from datasets import Dataset
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast
from torch import nn
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import argparse
import hashlib
import json
import os
import numpy as np
import pandas as pd
import torch


class UniModalClassifier(nn.Module):
    def __init__(self, config, num_classes, modality):
        super(UniModalClassifier, self).__init__()
        if modality not in ['ps', 'byte']:
            raise ValueError(f"Unsupported modality: {modality}")
        self.modality = modality
        self.encoder = BertForMaskedLM(config)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, inputs, return_features=False):
        if self.modality == 'ps':
            outputs = self.encoder.bert(input_ids=inputs['ps'], attention_mask=inputs['ps_attention_mask'])
        else:
            outputs = self.encoder.bert(
                input_ids=inputs['raw'],
                attention_mask=inputs['raw_attention_mask'],
                token_type_ids=inputs['raw_token_type_ids']
            )
        features = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(features)
        if return_features:
            return {'logits': logits, 'features': features}
        return {'logits': logits}


class MMClassifier(nn.Module):
    def __init__(self, ps_config, bytes_config, num_classes):
        super(MMClassifier, self).__init__()
        self.ps_encoder = BertForMaskedLM(ps_config)
        self.bytes_encoder = BertForMaskedLM(bytes_config)
        self.ps_cross_attention = nn.MultiheadAttention(embed_dim=ps_config.hidden_size, num_heads=4, batch_first=True)
        self.bytes_cross_attention = nn.MultiheadAttention(embed_dim=bytes_config.hidden_size, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(ps_config.hidden_size + bytes_config.hidden_size, num_classes)
        )

    def forward(self, inputs, return_features=False):
        ps_outputs = self.ps_encoder.bert(input_ids=inputs['ps'], attention_mask=inputs['ps_attention_mask'])
        raw_outputs = self.bytes_encoder.bert(
            input_ids=inputs['raw'],
            attention_mask=inputs['raw_attention_mask'],
            token_type_ids=inputs['raw_token_type_ids']
        )

        outputs = torch.concat([ps_outputs.last_hidden_state, raw_outputs.last_hidden_state], dim=1)
        key_padding_mask = (1 - torch.concat([inputs['ps_attention_mask'], inputs['raw_attention_mask']], dim=1)).bool()
        ps_attn_output, _ = self.ps_cross_attention(
            ps_outputs.last_hidden_state, outputs, outputs, key_padding_mask=key_padding_mask, need_weights=False
        )
        raw_attn_output, _ = self.bytes_cross_attention(
            raw_outputs.last_hidden_state, outputs, outputs, key_padding_mask=key_padding_mask, need_weights=False
        )

        memory_ps, memory_raw = ps_attn_output[:, 0, :], raw_attn_output[:, 0, :]
        features = torch.concat([memory_ps, memory_raw], dim=1)
        logits = self.classifier(features)
        if return_features:
            return {'logits': logits, 'features': features}
        return {'logits': logits}


device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_length_ps = 256
max_length_bytes = 512
min_pkts = 5
model_name = "MM4flow"
premodel_name_ps = 'BERT-ps'
premodel_name_raw = 'BERT-bytes'

tokenizer_ps = BertTokenizerFast.from_pretrained('tokenizer_bert/ps_tokenizer')
tokenizer_bytes = BertTokenizerFast.from_pretrained('tokenizer_bert/bytes_tokenizer')
drop_empty = lambda x: x if x != '(empty)' else np.nan


def parse_views(views_arg):
    views = [v.strip() for v in views_arg.split(',') if v.strip()]
    allowed = {'ps', 'byte', 'mm'}
    if not views:
        raise ValueError("--views must contain at least one view.")
    if len(set(views)) != len(views):
        raise ValueError(f"Duplicate views are not supported: {views}")
    unsupported = [v for v in views if v not in allowed]
    if unsupported:
        raise ValueError(f"Unsupported views: {unsupported}. Supported views: {sorted(allowed)}")
    return views


def label_hash(label2idx):
    payload = json.dumps(label2idx, sort_keys=True)
    return hashlib.md5(payload.encode('utf-8')).hexdigest()


def func_ps(ps):
    r = []
    for burst in str(ps).split(','):
        if not burst or burst == '(empty)':
            continue
        p_len, p_count = burst.split(':')
        r += [f"p{p_len}t"] * int(p_count)
        if len(r) > max_length_ps:
            break
    return ' '.join(r[0:max_length_ps])


def func_bytes(s):
    s = '' if pd.isna(s) else str(s)
    return ' '.join([s[i:i + 2] for i in range(0, len(s), 2)])


def preprocess_dataframe(df, modality, label2idx=None):
    df = df.copy()
    if 'up' in df.columns and 'down' in df.columns:
        df = df[df['up'] + df['down'] >= min_pkts].copy()

    if modality in ['ps', 'mm']:
        df['ps'] = df['ps'].apply(func_ps)
    if modality in ['byte', 'mm']:
        df['fwd_raw'] = df['fwd_raw'].apply(drop_empty).fillna(' ').apply(func_bytes)
        df['bwd_raw'] = df['bwd_raw'].apply(drop_empty).fillna(' ').apply(func_bytes)
    if modality not in ['ps', 'byte', 'mm']:
        raise ValueError(f"Unsupported modality: {modality}")

    if label2idx is not None and 'label' in df.columns:
        df['y_label'] = df['label'].map(label2idx)
    return df


def dataset_columns(modality, has_label):
    label_cols = ['y_label'] if has_label else []
    if modality == 'ps':
        return ['ps'] + label_cols
    if modality == 'byte':
        return ['fwd_raw', 'bwd_raw'] + label_cols
    return ['ps', 'fwd_raw', 'bwd_raw'] + label_cols


def tensor_columns(modality, has_label):
    label_cols = ['y_label'] if has_label else []
    if modality == 'ps':
        return ['ps', 'ps_attention_mask'] + label_cols
    if modality == 'byte':
        return ['raw', 'raw_attention_mask', 'raw_token_type_ids'] + label_cols
    return ['ps', 'ps_attention_mask', 'raw', 'raw_attention_mask', 'raw_token_type_ids'] + label_cols


def make_encoder(modality):
    def encode(examples):
        encoded = {}
        if modality in ['ps', 'mm']:
            ps = tokenizer_ps(
                examples['ps'],
                truncation=True,
                padding='max_length',
                max_length=max_length_ps,
                return_special_tokens_mask=True
            )
            encoded.update({'ps': ps['input_ids'], 'ps_attention_mask': ps['attention_mask']})

        if modality in ['byte', 'mm']:
            raw = tokenizer_bytes(
                list(zip(examples['fwd_raw'], examples['bwd_raw'])),
                truncation=True,
                padding='max_length',
                max_length=max_length_bytes,
                return_special_tokens_mask=True
            )
            encoded.update({
                'raw': raw['input_ids'],
                'raw_attention_mask': raw['attention_mask'],
                'raw_token_type_ids': raw['token_type_ids']
            })

        return encoded

    return encode


def make_dataset(df, modality, has_label):
    dataset = Dataset.from_pandas(df[dataset_columns(modality, has_label)])
    dataset = dataset.map(make_encoder(modality), batched=True)
    remove_columns = [col for col in ['fwd_raw', 'bwd_raw', 'uid', '__index_level_0__'] if col in dataset.column_names]
    dataset = dataset.remove_columns(remove_columns)
    dataset.set_format(type='torch', columns=tensor_columns(modality, has_label))
    return dataset


def load_model_info(model_ts):
    model_path = os.path.join('model-classifier', model_ts)
    with open(os.path.join(model_path, model_name, 'info.json')) as f:
        info = json.load(f)
    return model_path, info


def assert_model_infos_match(model_infos, views):
    base_view = views[0]
    base_label2idx = model_infos[base_view]['label2idx']
    for view in views:
        info = model_infos[view]
        if info['label2idx'] != base_label2idx:
            raise ValueError(f"{view} model label2idx mapping does not match {base_view}.")
        if info.get('modality') != view:
            raise ValueError(f"{view}_model_ts must point to a {view} model, got {info.get('modality')}")


def load_config(config_modality, info):
    if config_modality == 'ps':
        pre_timestamp = info['pre_timestamp_ps']
        with open(os.path.join('model', f'{premodel_name_ps}_{pre_timestamp}', 'hyperparameters.json')) as f:
            hyper = json.load(f)
        return BertConfig(
            vocab_size=len(tokenizer_ps.get_vocab()), max_position_embeddings=max_length_ps,
            hidden_size=hyper['d_model'], num_hidden_layers=hyper['n_layer'],
            num_attention_heads=hyper['n_head'], intermediate_size=hyper['dim_ff'],
        )

    pre_timestamp = info['pre_timestamp_raw']
    with open(os.path.join('model', f'{premodel_name_raw}_{pre_timestamp}', 'hyperparameters.json')) as f:
        hyper = json.load(f)
    return BertConfig(
        vocab_size=len(tokenizer_bytes.get_vocab()), max_position_embeddings=max_length_bytes,
        hidden_size=hyper['d_model'], num_hidden_layers=hyper['n_layer'],
        num_attention_heads=hyper['n_head'], intermediate_size=hyper['dim_ff'],
    )


def classifier_num_classes_from_state(state_dict, modality):
    if modality == 'mm':
        weight = state_dict.get('classifier.0.weight')
    else:
        weight = state_dict.get('classifier.weight')
    if weight is None:
        raise ValueError(f"Missing classifier weight for {modality} checkpoint.")
    return weight.shape[0]


def load_view_model(model_path, info, view, num_classes):
    model_type = info.get('model_type', f'{view}-finetune')
    model_dir = os.path.join(model_path, model_name, model_type)
    state_path = os.path.join(model_dir, 'pytorch_model.bin')
    state_dict = torch.load(state_path, map_location=device)
    checkpoint_num_classes = classifier_num_classes_from_state(state_dict, view)
    if checkpoint_num_classes != num_classes:
        raise ValueError(
            f"Label/checkpoint mismatch for {view} model.\n"
            f"info.json has {num_classes} classes, but checkpoint classifier has {checkpoint_num_classes} classes.\n"
            f"model_path={model_path}\n"
            f"state_path={state_path}\n"
            f"Likely causes: wrong --{view}_model_ts, reused OUTPUT_NAME/RUN_TS, or info.json overwritten by another dataset."
        )

    if view == 'mm':
        model = MMClassifier(
            ps_config=load_config('ps', info),
            bytes_config=load_config('byte', info),
            num_classes=num_classes,
        ).to(device)
    else:
        model = UniModalClassifier(load_config(view, info), num_classes=num_classes, modality=view).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def collect_outputs(model, dataset, batch_size, desc):
    logits_list, probs_list, features_list = [], [], []
    with torch.inference_mode():
        for i in tqdm(range(0, dataset.num_rows, batch_size), desc=desc):
            batch_inputs = {k: v.to(device) for k, v in dataset[i:i + batch_size].items() if k != 'y_label'}
            output = model(batch_inputs, return_features=True)
            logits = output['logits'].float().cpu()
            probs = torch.softmax(logits, dim=-1)
            logits_list.append(logits.numpy())
            probs_list.append(probs.numpy())
            features_list.append(output['features'].float().cpu().numpy())

    logits = np.concatenate(logits_list, axis=0)
    probs = np.concatenate(probs_list, axis=0)
    features = np.concatenate(features_list, axis=0)
    return {'logits': logits, 'probs': probs, 'features': features, 'pred': probs.argmax(axis=1)}


def cache_entry(split, view, dataset_path, model_ts, num_classes, label2idx, sample_count):
    return {
        'split': split,
        'view': view,
        'dataset_path': dataset_path,
        'model_ts': model_ts,
        'num_classes': int(num_classes),
        'label_hash': label_hash(label2idx),
        'sample_count': int(sample_count),
    }


def load_cache_meta(cache_dir):
    path = os.path.join(cache_dir, 'cache_meta.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_cache_meta(cache_dir, meta):
    with open(os.path.join(cache_dir, 'cache_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def cache_matches(actual, expected):
    return actual == expected


def try_load_outputs_from_cache(split, view, dataset_path, model_ts, num_classes, label2idx, sample_count, args):
    cache_dir = args.cache_dir or os.path.join(args.output_dir, 'cache')
    key = f'{split}_{view}'
    cache_path = os.path.join(cache_dir, f'{key}.npz')
    expected_meta = cache_entry(split, view, dataset_path, model_ts, num_classes, label2idx, sample_count)
    meta = load_cache_meta(cache_dir)
    if args.overwrite_cache or not os.path.exists(cache_path) or not cache_matches(meta.get(key), expected_meta):
        return None
    print(f"[cache-hit] {key}: {cache_path}")
    cached = np.load(cache_path)
    return {name: cached[name] for name in ['logits', 'probs', 'features', 'pred']}


def get_outputs_with_cache(model, dataset, split, view, dataset_path, model_ts, num_classes, label2idx, args):
    cache_dir = args.cache_dir or os.path.join(args.output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    key = f'{split}_{view}'
    cache_path = os.path.join(cache_dir, f'{key}.npz')
    expected_meta = cache_entry(split, view, dataset_path, model_ts, num_classes, label2idx, dataset.num_rows)
    meta = load_cache_meta(cache_dir)

    if not args.overwrite_cache and os.path.exists(cache_path) and cache_matches(meta.get(key), expected_meta):
        print(f"[cache-hit] {key}: {cache_path}")
        cached = np.load(cache_path)
        return {name: cached[name] for name in ['logits', 'probs', 'features', 'pred']}

    outputs = collect_outputs(model, dataset, args.batch_size, f'{split} {view}')
    np.savez_compressed(
        cache_path,
        logits=outputs['logits'],
        probs=outputs['probs'],
        features=outputs['features'],
        pred=outputs['pred'],
    )
    meta[key] = expected_meta
    save_cache_meta(cache_dir, meta)
    print(f"[cache-save] {key}: {cache_path}")
    return outputs


def safe_accuracy(y_true, y_pred, mask=None):
    if mask is None:
        mask = np.ones_like(y_true, dtype=bool)
    if mask.sum() == 0:
        return None
    return float(accuracy_score(y_true[mask], y_pred[mask]))


def per_class_recall(y_true, y_pred, num_classes, mask):
    recalls = np.zeros(num_classes, dtype=np.float64)
    supports = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        c_mask = mask & (y_true == c)
        supports[c] = int(c_mask.sum())
        if supports[c] > 0:
            recalls[c] = float((y_pred[c_mask] == c).mean())
        else:
            recalls[c] = np.nan
    return recalls, supports


def class_oracle_predictions(y_true, preds_by_view, views, num_classes, mask):
    recall_by_view = {}
    supports = None
    for view in views:
        recalls, view_supports = per_class_recall(y_true, preds_by_view[view], num_classes, mask)
        recall_by_view[view] = recalls
        supports = view_supports if supports is None else supports

    recall_matrix = np.stack([np.nan_to_num(recall_by_view[view], nan=-1.0) for view in views], axis=0)
    best_indices = recall_matrix.argmax(axis=0)
    best_view = np.array([views[i] for i in best_indices], dtype=object)
    y_oracle = preds_by_view[views[0]].copy()
    known_idx = np.where(mask)[0]
    for idx in known_idx:
        y_oracle[idx] = preds_by_view[best_view[y_true[idx]]][idx]
    return y_oracle, best_view, recall_by_view, supports


def sample_oracle_predictions(y_true, preds_by_view, views):
    y_oracle = preds_by_view[views[0]].copy()
    for view in views[1:]:
        first_wrong_view_right = (y_oracle != y_true) & (preds_by_view[view] == y_true)
        y_oracle[first_wrong_view_right] = preds_by_view[view][first_wrong_view_right]
    return y_oracle


def robust_zscore(values, eps=1e-6):
    values = values.astype(np.float64)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        scale = np.nanstd(values)
    if not np.isfinite(scale) or scale < eps:
        return np.zeros_like(values, dtype=np.float64)
    return (values - median) / (scale + eps)


def softmax_np(values, axis=0, tau=1.0):
    scaled = values / max(tau, 1e-8)
    scaled = scaled - np.nanmax(scaled, axis=axis, keepdims=True)
    exp = np.exp(scaled)
    exp = np.where(np.isfinite(exp), exp, 0.0)
    denom = exp.sum(axis=axis, keepdims=True)
    return exp / np.maximum(denom, 1e-12)


def aggregate_values(values, method):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    if method == 'mean':
        return float(np.mean(values))
    if method == 'median':
        return float(np.median(values))
    if method == 'trimmed_mean':
        if len(values) < 5:
            return float(np.mean(values))
        values = np.sort(values)
        k = max(1, int(0.1 * len(values)))
        if 2 * k >= len(values):
            return float(np.mean(values))
        return float(np.mean(values[k:-k]))
    raise ValueError(f"Unsupported aggregate: {method}")


def source_priors(y_source, num_classes):
    counts = np.bincount(y_source, minlength=num_classes).astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def penalties(pred, num_classes, source_prior, w_prior, w_collapse, w_support):
    pred_counts = np.bincount(pred, minlength=num_classes).astype(np.float64)
    pred_prior = pred_counts / max(pred_counts.sum(), 1.0)
    expected_support = 1.0 / max(num_classes, 1)
    low_support = np.maximum(0.0, expected_support - pred_prior) / max(expected_support, 1e-12)
    prior_shift = np.abs(pred_prior - source_prior)
    collapse = np.maximum(0.0, pred_prior - source_prior)
    return w_prior * prior_shift + w_collapse * collapse + w_support * low_support


def prob_only_evidence(probs, pred, num_classes, source_prior, args):
    h = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        mask = pred == c
        if mask.any():
            h[c] = float(np.mean(probs[mask, c]))
        else:
            h[c] = 0.0
    return h - penalties(pred, num_classes, source_prior, args.w_prior, args.w_collapse, args.w_support)


def fit_diagonal_gaussians(features, labels, num_classes, var_eps):
    dim = features.shape[1]
    means = np.zeros((num_classes, dim), dtype=np.float64)
    variances = np.ones((num_classes, dim), dtype=np.float64)
    global_var = np.var(features, axis=0) + var_eps
    for c in range(num_classes):
        c_features = features[labels == c]
        if len(c_features) == 0:
            means[c] = np.mean(features, axis=0)
            variances[c] = global_var
        elif len(c_features) == 1:
            means[c] = c_features[0]
            variances[c] = global_var
        else:
            means[c] = np.mean(c_features, axis=0)
            variances[c] = np.var(c_features, axis=0) + var_eps
    return means, variances


def mahalanobis_distances(features, means, variances):
    distances = []
    for c in range(means.shape[0]):
        diff = features - means[c]
        distances.append(np.mean((diff * diff) / variances[c], axis=1))
    return np.stack(distances, axis=1)


def geometry_evidence(source_features, source_labels, target_features, pred, num_classes, source_prior, args):
    means, variances = fit_diagonal_gaussians(source_features, source_labels, num_classes, args.var_eps)
    distances = mahalanobis_distances(target_features, means, variances)
    neg_dist_z = np.zeros_like(distances)
    margin_z = np.zeros_like(distances)

    for c in range(num_classes):
        other = np.delete(distances, c, axis=1)
        nearest_other = np.min(other, axis=1)
        margin = nearest_other - distances[:, c]
        neg_dist_z[:, c] = robust_zscore(-distances[:, c])
        margin_z[:, c] = robust_zscore(margin)

    sample_evidence = neg_dist_z + margin_z
    h = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        mask = pred == c
        h[c] = aggregate_values(sample_evidence[mask, c], args.aggregate) if mask.any() else 0.0

    h = h - penalties(pred, num_classes, source_prior, args.w_prior, args.w_collapse, args.w_support)
    return h, distances, sample_evidence


def fuse_probs(weights, probs_by_view, views):
    fused = np.zeros_like(probs_by_view[views[0]], dtype=np.float64)
    for i, view in enumerate(views):
        fused += weights[i][None, :] * probs_by_view[view]
    fused = fused / np.maximum(fused.sum(axis=1, keepdims=True), 1e-12)
    return fused


def corr_ignore_nan(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return None
    if np.std(a[mask]) < 1e-12 or np.std(b[mask]) < 1e-12:
        return None
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def write_weights(path, weights, h_values, idx2label, views):
    rows = []
    for c, label in idx2label.items():
        row = {'class_idx': c, 'label': label}
        for i, view in enumerate(views):
            row[f'H_{view}'] = h_values[i, c]
        for i, view in enumerate(views):
            row[f'W_{view}'] = weights[i, c]
        row['selected_view'] = views[int(np.argmax(weights[:, c]))]
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def build_classification_reports(y_true, predictions, idx2label, known_mask):
    labels = list(idx2label.keys())
    target_names = [idx2label[i] for i in labels]
    report_texts = {}
    report_dicts = {}
    report_rows = []

    for name, y_pred in predictions.items():
        text = classification_report(
            y_true[known_mask],
            y_pred[known_mask],
            labels=labels,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
        report = classification_report(
            y_true[known_mask],
            y_pred[known_mask],
            labels=labels,
            target_names=target_names,
            digits=4,
            zero_division=0,
            output_dict=True,
        )
        report_texts[name] = text
        report_dicts[name] = report

        for label_name, metrics in report.items():
            if isinstance(metrics, dict):
                row = {'method': name, 'label': label_name}
                row.update(metrics)
                report_rows.append(row)
            else:
                report_rows.append({'method': name, 'label': label_name, 'accuracy': metrics})

    return report_texts, report_dicts, pd.DataFrame(report_rows)


def write_classification_reports(output_dir, report_texts, report_dicts, report_df):
    with open(os.path.join(output_dir, 'classification_reports.txt'), 'w') as f:
        for name, text in report_texts.items():
            f.write(f"===== {name} =====\n")
            f.write(text)
            f.write("\n\n")
    with open(os.path.join(output_dir, 'classification_reports.json'), 'w') as f:
        json.dump(report_dicts, f, indent=2)
    report_df.to_csv(os.path.join(output_dir, 'classification_reports.csv'), index=False)


def idx2label_from_weight_file(weight_path):
    weights = pd.read_csv(weight_path)
    idx2label = dict(zip(weights['class_idx'].astype(int), weights['label'].astype(str)))
    return dict(sorted(idx2label.items(), key=lambda item: item[0]))


def report_only_from_existing_outputs(output_dir):
    pred_path = os.path.join(output_dir, 'target_predictions.csv')
    weight_path = os.path.join(output_dir, 'class_weights_prob.csv')
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing {pred_path}; run full fusion first or pass the correct --output_dir.")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Missing {weight_path}; it is used to recover class_idx -> label mapping.")

    target_out = pd.read_csv(pred_path)
    idx2label = idx2label_from_weight_file(weight_path)

    y_true = target_out['y_true_idx'].fillna(-1).astype(int).to_numpy()
    known_mask = y_true >= 0
    predictions = {}
    report_name_map = {
        'avg': 'avg_prob',
        'prob_only': 'prob_only_class_fusion',
        'geometry': 'geometry_class_fusion',
    }
    for column in target_out.columns:
        if column.startswith('pred_') and column.endswith('_idx'):
            name = column[len('pred_'):-len('_idx')]
            predictions[report_name_map.get(name, name)] = target_out[column].astype(int).to_numpy()

    view_names = [name for name in predictions if name not in {
        'avg_prob', 'prob_only_class_fusion', 'geometry_class_fusion', 'class_oracle', 'sample_oracle'
    }]
    if 'class_oracle' not in predictions and view_names:
        y_class_oracle, _, _, _ = class_oracle_predictions(
            y_true, predictions, view_names, len(idx2label), known_mask
        )
        predictions['class_oracle'] = y_class_oracle
    if 'sample_oracle' not in predictions and view_names:
        predictions['sample_oracle'] = sample_oracle_predictions(y_true, predictions, view_names)

    report_texts, report_dicts, report_df = build_classification_reports(
        y_true, predictions, idx2label, known_mask
    )
    write_classification_reports(output_dir, report_texts, report_dicts, report_df)
    print(f"saved classification reports under: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataset')
    parser.add_argument('--target_csv')
    parser.add_argument('--ps_model_ts')
    parser.add_argument('--byte_model_ts')
    parser.add_argument('--mm_model_ts')
    parser.add_argument('--views', default='ps,byte')
    parser.add_argument('--cache_dir', default=None)
    parser.add_argument('--overwrite_cache', action='store_true')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--aggregate', choices=['mean', 'median', 'trimmed_mean'], default='mean')
    parser.add_argument('--w_prior', type=float, default=0.5)
    parser.add_argument('--w_collapse', type=float, default=0.5)
    parser.add_argument('--w_support', type=float, default=0.2)
    parser.add_argument('--var_eps', type=float, default=1e-4)
    parser.add_argument('--max_source_samples', type=int, default=None)
    parser.add_argument('--max_target_samples', type=int, default=None)
    parser.add_argument(
        '--report_only',
        action='store_true',
        help='Only regenerate classification_reports.* from an existing output_dir/target_predictions.csv.',
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.report_only:
        report_only_from_existing_outputs(args.output_dir)
        return

    views = parse_views(args.views)
    required_args = ['source_dataset', 'target_csv'] + [f'{view}_model_ts' for view in views]
    missing_args = [name for name in required_args if getattr(args, name) is None]
    if missing_args:
        raise ValueError(f"Missing required arguments for full run: {', '.join(missing_args)}")

    model_paths, model_infos, model_ts_by_view = {}, {}, {}
    for view in views:
        model_ts = getattr(args, f'{view}_model_ts')
        model_ts_by_view[view] = model_ts
        model_paths[view], model_infos[view] = load_model_info(model_ts)
    assert_model_infos_match(model_infos, views)

    label2idx = model_infos[views[0]]['label2idx']
    idx2label = {idx: label for label, idx in label2idx.items()}
    idx2label = dict(sorted(idx2label.items(), key=lambda item: item[0]))
    num_classes = len(label2idx)

    source_df = pd.read_csv(os.path.join(args.source_dataset, 'train.csv.gz'), compression='gzip', index_col=0)
    target_df = pd.read_csv(args.target_csv, compression='gzip', index_col=0)
    if args.max_source_samples is not None:
        source_df = source_df.head(args.max_source_samples)
    if args.max_target_samples is not None:
        target_df = target_df.head(args.max_target_samples)

    source_view_dfs = {view: preprocess_dataframe(source_df, view, label2idx) for view in views}
    target_view_dfs = {view: preprocess_dataframe(target_df, view, label2idx) for view in views}

    source_len = len(source_view_dfs[views[0]])
    target_len = len(target_view_dfs[views[0]])
    for view in views[1:]:
        if len(source_view_dfs[view]) != source_len:
            raise ValueError(f"Source sample mismatch after preprocessing: {views[0]}={source_len}, {view}={len(source_view_dfs[view])}")
        if len(target_view_dfs[view]) != target_len:
            raise ValueError(f"Target sample mismatch after preprocessing: {views[0]}={target_len}, {view}={len(target_view_dfs[view])}")

    source_mask = source_view_dfs[views[0]]['y_label'].notna().to_numpy()
    target_known_mask = (
        target_view_dfs[views[0]]['y_label'].notna().to_numpy()
        if 'y_label' in target_view_dfs[views[0]].columns
        else np.zeros(target_len, dtype=bool)
    )
    y_source = source_view_dfs[views[0]]['y_label'].fillna(-1).astype(int).to_numpy()
    y_target = target_view_dfs[views[0]]['y_label'].fillna(-1).astype(int).to_numpy()

    source_datasets = {view: make_dataset(source_view_dfs[view], view, has_label=True) for view in views}
    target_has_label = 'label' in target_view_dfs[views[0]].columns
    target_datasets = {view: make_dataset(target_view_dfs[view], view, has_label=target_has_label) for view in views}

    source_outputs, target_outputs = {}, {}
    for view in views:
        source_cached = try_load_outputs_from_cache(
            'source', view, args.source_dataset, model_ts_by_view[view],
            num_classes, label2idx, source_datasets[view].num_rows, args
        )
        target_cached = try_load_outputs_from_cache(
            'target', view, args.target_csv, model_ts_by_view[view],
            num_classes, label2idx, target_datasets[view].num_rows, args
        )
        if source_cached is not None and target_cached is not None:
            source_outputs[view] = source_cached
            target_outputs[view] = target_cached
            continue

        model = load_view_model(model_paths[view], model_infos[view], view, num_classes)
        source_outputs[view] = get_outputs_with_cache(
            model, source_datasets[view], 'source', view, args.source_dataset,
            model_ts_by_view[view], num_classes, label2idx, args
        )
        target_outputs[view] = get_outputs_with_cache(
            model, target_datasets[view], 'target', view, args.target_csv,
            model_ts_by_view[view], num_classes, label2idx, args
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    target_probs = {view: target_outputs[view]['probs'] for view in views}
    target_preds = {view: target_outputs[view]['pred'].astype(int) for view in views}
    pred_avg = np.mean([target_probs[view] for view in views], axis=0).argmax(axis=1)

    y_class_oracle, class_best_view, recall_by_view, target_supports = class_oracle_predictions(
        y_target, target_preds, views, num_classes, target_known_mask
    )
    y_sample_oracle = sample_oracle_predictions(y_target, target_preds, views)

    src_prior = source_priors(y_source[source_mask], num_classes)
    h_prob = np.stack([
        prob_only_evidence(target_probs[view], target_preds[view], num_classes, src_prior, args)
        for view in views
    ], axis=0)
    weights_prob = softmax_np(h_prob, axis=0, tau=args.tau)
    probs_prob = fuse_probs(weights_prob, target_probs, views)
    pred_prob = probs_prob.argmax(axis=1)

    h_geo = np.stack([
        geometry_evidence(
            source_outputs[view]['features'][source_mask],
            y_source[source_mask],
            target_outputs[view]['features'],
            target_preds[view],
            num_classes,
            src_prior,
            args
        )[0]
        for view in views
    ], axis=0)
    weights_geo = softmax_np(h_geo, axis=0, tau=args.tau)
    probs_geo = fuse_probs(weights_geo, target_probs, views)
    pred_geo = probs_geo.argmax(axis=1)

    prob_selected = np.array([views[i] for i in weights_prob.argmax(axis=0)], dtype=object)
    geo_selected = np.array([views[i] for i in weights_geo.argmax(axis=0)], dtype=object)
    health_best_view_match_rate_prob = float((prob_selected == class_best_view).mean())
    health_best_view_match_rate_geo = float((geo_selected == class_best_view).mean())

    per_rows = []
    for c, label in idx2label.items():
        row = {
            'class_idx': c,
            'label': label,
            'support': int(target_supports[c]),
            'best_recall_view': class_best_view[c],
        }
        for view in views:
            row[f'{view}_recall'] = recall_by_view[view][c]
        for i, view in enumerate(views):
            row[f'H_prob_{view}'] = h_prob[i, c]
        for i, view in enumerate(views):
            row[f'W_prob_{view}'] = weights_prob[i, c]
        row['prob_selected_view'] = prob_selected[c]
        for i, view in enumerate(views):
            row[f'H_geo_{view}'] = h_geo[i, c]
        for i, view in enumerate(views):
            row[f'W_geo_{view}'] = weights_geo[i, c]
        row['geo_selected_view'] = geo_selected[c]
        per_rows.append(row)
    per_class_df = pd.DataFrame(per_rows)

    h_prob_best = np.array([h_prob[views.index(class_best_view[c]), c] for c in range(num_classes)])
    h_geo_best = np.array([h_geo[views.index(class_best_view[c]), c] for c in range(num_classes)])
    best_recall = np.array([recall_by_view[class_best_view[c]][c] for c in range(num_classes)])

    acc_by_view = {f'{view}_acc': safe_accuracy(y_target, target_preds[view], target_known_mask) for view in views}
    best_single_acc = max((acc or 0.0) for acc in acc_by_view.values())
    summary = {
        'source_dataset': args.source_dataset,
        'target_csv': args.target_csv,
        'views': views,
        'model_ts_by_view': model_ts_by_view,
        'cache_dir': args.cache_dir or os.path.join(args.output_dir, 'cache'),
        'num_classes': num_classes,
        'num_source_samples': int(source_len),
        'num_target_samples': int(target_len),
        'num_target_known_label_samples': int(target_known_mask.sum()),
        **acc_by_view,
        'best_single_acc': best_single_acc,
        'avg_prob_acc': safe_accuracy(y_target, pred_avg, target_known_mask),
        'prob_only_class_fusion_acc': safe_accuracy(y_target, pred_prob, target_known_mask),
        'geometry_class_fusion_acc': safe_accuracy(y_target, pred_geo, target_known_mask),
        'class_oracle_acc': safe_accuracy(y_target, y_class_oracle, target_known_mask),
        'sample_oracle_acc': safe_accuracy(y_target, y_sample_oracle, target_known_mask),
        'health_best_view_match_rate_prob': health_best_view_match_rate_prob,
        'health_best_view_match_rate_geo': health_best_view_match_rate_geo,
        'corr_h_prob_selected_with_best_recall': corr_ignore_nan(h_prob_best, best_recall),
        'corr_h_geo_selected_with_best_recall': corr_ignore_nan(h_geo_best, best_recall),
        'warnings': [],
    }
    if summary['sample_oracle_acc'] is not None and summary['sample_oracle_acc'] + 1e-12 < summary['best_single_acc']:
        summary['warnings'].append('sample_oracle_acc is lower than best_single_acc; check labels/predictions.')
    if summary['class_oracle_acc'] is not None and summary['class_oracle_acc'] + 1e-12 < summary['best_single_acc']:
        summary['warnings'].append('class_oracle_acc is lower than best_single_acc; this may happen if target has missing classes.')

    target_out = target_view_dfs[views[0]].copy()
    target_out['y_true_idx'] = y_target
    for view in views:
        target_out[f'pred_{view}_idx'] = target_preds[view]
    target_out['pred_avg_idx'] = pred_avg
    target_out['pred_prob_only_idx'] = pred_prob
    target_out['pred_geometry_idx'] = pred_geo
    target_out['pred_class_oracle_idx'] = y_class_oracle
    target_out['pred_sample_oracle_idx'] = y_sample_oracle
    for view in views:
        target_out[f'pred_{view}'] = [idx2label[i] for i in target_preds[view]]
    target_out['pred_avg'] = [idx2label[i] for i in pred_avg]
    target_out['pred_prob_only'] = [idx2label[i] for i in pred_prob]
    target_out['pred_geometry'] = [idx2label[i] for i in pred_geo]
    target_out['pred_class_oracle'] = [idx2label[i] for i in y_class_oracle]
    target_out['pred_sample_oracle'] = [idx2label[i] for i in y_sample_oracle]
    for view in views:
        for c, label in idx2label.items():
            target_out[f'prob_{view}_{label}'] = target_probs[view][:, c]

    report_predictions = {view: target_preds[view] for view in views}
    report_predictions.update({
        'avg_prob': pred_avg,
        'prob_only_class_fusion': pred_prob,
        'geometry_class_fusion': pred_geo,
        'class_oracle': y_class_oracle,
        'sample_oracle': y_sample_oracle,
    })
    report_texts, report_dicts, report_df = build_classification_reports(
        y_target,
        report_predictions,
        idx2label,
        target_known_mask,
    )

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    per_class_df.to_csv(os.path.join(args.output_dir, 'per_class_metrics.csv'), index=False)
    target_out.to_csv(os.path.join(args.output_dir, 'target_predictions.csv'), index=False)
    write_weights(os.path.join(args.output_dir, 'class_weights_prob.csv'), weights_prob, h_prob, idx2label, views)
    write_weights(os.path.join(args.output_dir, 'class_weights_geometry.csv'), weights_geo, h_geo, idx2label, views)
    write_classification_reports(args.output_dir, report_texts, report_dicts, report_df)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
