from datasets import Dataset
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast
from torch import nn
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
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


def entropy_rows(probs):
    probs = np.asarray(probs, dtype=np.float64)
    return -np.sum(probs * np.log(probs + 1e-12), axis=1)


def entropy_dist(dist):
    dist = np.asarray(dist, dtype=np.float64)
    dist = dist / max(float(dist.sum()), 1e-12)
    return float(-np.sum(dist * np.log(dist + 1e-12)))


def js_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    m = 0.5 * (p + q)
    return float(0.5 * (
        np.sum(p * (np.log(p + 1e-12) - np.log(m + 1e-12))) +
        np.sum(q * (np.log(q + 1e-12) - np.log(m + 1e-12)))
    ))


def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    return float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))


def pred_frequency(pred, num_classes):
    counts = np.bincount(np.asarray(pred, dtype=int), minlength=num_classes).astype(np.float64)
    return counts / max(float(counts.sum()), 1e-12)


def rank_likelihood(values, higher_better=False, power=1.0):
    values = np.asarray(values, dtype=np.float64)
    finite = np.where(np.isfinite(values), values, -np.inf if higher_better else np.inf)
    order = np.argsort(-finite if higher_better else finite, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    likelihood = (len(values) + 1.0 - ranks) / max(float(len(values)), 1.0)
    return np.power(likelihood, power)


def margin_mean(probs):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape[1] < 2:
        return float(np.mean(probs[:, 0]))
    part = np.partition(probs, -2, axis=1)
    return float(np.mean(part[:, -1] - part[:, -2]))


def hspf_make_subset_key(subset):
    return '+'.join(subset)


def hspf_candidate_name(subset, rule):
    return f"{hspf_make_subset_key(subset)}__{rule}"


def hspf_fuse_subset(subset, rule, probs_by_view, weights_prob, weights_geo, weights_agreement, views):
    subset = list(subset)
    if rule == 'identity':
        return probs_by_view[subset[0]]
    if rule == 'average':
        return np.mean([probs_by_view[view] for view in subset], axis=0)
    view_indices = [views.index(view) for view in subset]
    if rule == 'prob_class':
        weights = weights_prob[view_indices, :]
    elif rule == 'geometry_class':
        weights = weights_geo[view_indices, :]
    elif rule == 'agreement_gate':
        weights = weights_agreement[view_indices, :]
    else:
        raise ValueError(f"Unsupported HSPF rule: {rule}")
    weights = weights / np.maximum(weights.sum(axis=0, keepdims=True), 1e-12)
    fused = np.zeros_like(probs_by_view[subset[0]], dtype=np.float64)
    for i, view in enumerate(subset):
        fused += weights[i][None, :] * probs_by_view[view]
    return fused / np.maximum(fused.sum(axis=1, keepdims=True), 1e-12)


def hspf_build_candidates(views, probs_by_view, weights_prob, weights_geo, weights_agreement):
    from itertools import combinations

    candidates = []
    for view in views:
        candidates.append({
            'name': hspf_candidate_name([view], 'identity'),
            'subset': (view,),
            'rule': 'identity',
            'probs': probs_by_view[view],
        })
    for size in range(2, len(views) + 1):
        for subset in combinations(views, size):
            for rule in ['average', 'prob_class', 'geometry_class', 'agreement_gate']:
                candidates.append({
                    'name': hspf_candidate_name(subset, rule),
                    'subset': tuple(subset),
                    'rule': rule,
                    'probs': hspf_fuse_subset(subset, rule, probs_by_view, weights_prob, weights_geo, weights_agreement, views),
                })
    return candidates


def hspf_weighted_identity_base(subset, probs_by_view, view_gate):
    weights = np.array([view_gate[view] for view in subset], dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)
    base = np.zeros_like(probs_by_view[subset[0]], dtype=np.float64)
    for weight, view in zip(weights, subset):
        base += weight * probs_by_view[view]
    return base / np.maximum(base.sum(axis=1, keepdims=True), 1e-12)


def hspf_source_scores(candidates, source_val_probs_by_view, source_val_labels, source_val_mask,
                       weights_prob, weights_geo, weights_agreement, views):
    scores = {}
    for cand in candidates:
        probs = hspf_fuse_subset(
            cand['subset'], cand['rule'], source_val_probs_by_view,
            weights_prob, weights_geo, weights_agreement, views
        )
        pred = probs.argmax(axis=1)
        if source_val_mask.sum() == 0:
            scores[cand['name']] = 0.0
            continue
        _, _, f1, _ = precision_recall_fscore_support(
            source_val_labels[source_val_mask],
            pred[source_val_mask],
            average='weighted',
            zero_division=0,
        )
        scores[cand['name']] = float(f1)
    return scores


def hspf_view_gate(views, source_outputs, target_outputs, num_classes, source_prior):
    evidence = {view: {} for view in views}
    for view in views:
        src_prob_mean = np.mean(source_outputs[view]['probs'], axis=0)
        tgt_prob_mean = np.mean(target_outputs[view]['probs'], axis=0)
        src_pred_freq = pred_frequency(source_outputs[view]['pred'], num_classes)
        tgt_pred_freq = pred_frequency(target_outputs[view]['pred'], num_classes)
        evidence[view]['prior_penalty'] = kl_divergence(tgt_prob_mean, source_prior)
        evidence[view]['source_target_pred_freq_js'] = js_divergence(src_pred_freq, tgt_pred_freq)
        evidence[view]['source_target_mean_prob_js'] = js_divergence(src_prob_mean, tgt_prob_mean)
        evidence[view]['collapse_penalty'] = float(np.max(tgt_pred_freq))
        evidence[view]['target_prediction_diversity_risk'] = 1.0 - entropy_dist(tgt_pred_freq) / max(np.log(num_classes), 1e-12)

    strengths = np.ones(len(views), dtype=np.float64)
    for metric in [
        'prior_penalty',
        'source_target_pred_freq_js',
        'source_target_mean_prob_js',
        'collapse_penalty',
        'target_prediction_diversity_risk',
    ]:
        values = np.array([evidence[view][metric] for view in views], dtype=np.float64)
        strengths *= rank_likelihood(values, higher_better=False)
    strengths = strengths / max(float(strengths.sum()), 1e-12)
    return {view: float(strengths[i]) for i, view in enumerate(views)}, evidence


def hspf_run(candidates, views, target_probs_by_view, source_outputs, target_outputs, source_prior,
             source_val_probs_by_view, source_val_labels, source_val_mask,
             weights_prob, weights_geo, weights_agreement, num_classes):
    view_gate, view_evidence = hspf_view_gate(views, source_outputs, target_outputs, num_classes, source_prior)
    source_scores = hspf_source_scores(
        candidates, source_val_probs_by_view, source_val_labels, source_val_mask,
        weights_prob, weights_geo, weights_agreement, views
    )

    subset_g, bal_risk, im_scores, harms, src_scores = [], [], [], [], []
    for cand in candidates:
        probs = cand['probs']
        subset = cand['subset']
        subset_gate = np.array([view_gate[view] for view in subset], dtype=np.float64)
        subset_g.append(float(np.prod(np.maximum(subset_gate, 1e-12)) ** (1.0 / len(subset))))
        mean_prob = np.mean(probs, axis=0)
        bal_risk.append(js_divergence(mean_prob, source_prior))
        im_scores.append(entropy_dist(mean_prob) - float(np.mean(entropy_rows(probs))))
        src_scores.append(source_scores[cand['name']])
        if cand['rule'] == 'identity':
            harms.append(0.0)
        else:
            base = hspf_weighted_identity_base(subset, target_probs_by_view, view_gate)
            ent_delta = max(float(np.mean(entropy_rows(probs)) - np.mean(entropy_rows(base))), 0.0)
            margin_delta = max(margin_mean(base) - margin_mean(probs), 0.0)
            flip = float(np.mean((probs.argmax(axis=1) != base.argmax(axis=1)) * np.max(base, axis=1)))
            balance_delta = max(js_divergence(mean_prob, source_prior) - js_divergence(np.mean(base, axis=0), source_prior), 0.0)
            harms.append(ent_delta + margin_delta + flip + balance_delta)

    subset_g = np.asarray(subset_g, dtype=np.float64)
    bal_risk = np.asarray(bal_risk, dtype=np.float64)
    im_scores = np.asarray(im_scores, dtype=np.float64)
    harms = np.asarray(harms, dtype=np.float64)
    src_scores = np.asarray(src_scores, dtype=np.float64)

    L_view = rank_likelihood(subset_g, higher_better=True, power=0.5)
    L_bal = rank_likelihood(bal_risk, higher_better=False)
    L_im = rank_likelihood(im_scores, higher_better=True)
    L_src = rank_likelihood(src_scores, higher_better=True, power=0.3)
    L_harm = rank_likelihood(harms, higher_better=False, power=2.0)
    L_c = L_bal * L_im * L_src * L_harm

    rules = ['identity', 'average', 'prob_class', 'geometry_class', 'agreement_gate']
    rule_scores = {}
    for rule in rules:
        values = np.array([L_c[i] for i, cand in enumerate(candidates) if cand['rule'] == rule], dtype=np.float64)
        if len(values) == 0:
            rule_scores[rule] = 0.0
        else:
            sorted_values = np.sort(values)[::-1]
            top_n = max(1, int(np.ceil(len(sorted_values) / 2.0)))
            rule_scores[rule] = float(np.mean(sorted_values[:top_n]))
    rule_total = sum(rule_scores.values())
    rule_prior = (
        {rule: 1.0 / len(rules) for rule in rules}
        if rule_total <= 0 else
        {rule: float(score / rule_total) for rule, score in rule_scores.items()}
    )

    L_rule = np.array([rule_prior[cand['rule']] ** 0.5 for cand in candidates], dtype=np.float64)
    posterior_raw = L_view * L_rule * L_c
    posterior = posterior_raw / max(float(posterior_raw.sum()), 1e-12)
    if not np.isfinite(posterior).all() or posterior.sum() <= 0:
        posterior = np.ones(len(candidates), dtype=np.float64) / len(candidates)

    final_probs = np.zeros_like(candidates[0]['probs'], dtype=np.float64)
    for weight, cand in zip(posterior, candidates):
        final_probs += weight * cand['probs']
    final_probs = final_probs / np.maximum(final_probs.sum(axis=1, keepdims=True), 1e-12)

    rows = []
    for rank, idx in enumerate(np.argsort(-posterior), start=1):
        cand = candidates[idx]
        rows.append({
            'rank': rank,
            'candidate': cand['name'],
            'subset': hspf_make_subset_key(cand['subset']),
            'rule': cand['rule'],
            'posterior': float(posterior[idx]),
            'L_view': float(L_view[idx]),
            'L_rule': float(L_rule[idx]),
            'L_bal': float(L_bal[idx]),
            'L_im': float(L_im[idx]),
            'L_src': float(L_src[idx]),
            'L_harm': float(L_harm[idx]),
            'balance_risk': float(bal_risk[idx]),
            'information_maximization': float(im_scores[idx]),
            'source_val_weighted_f1': float(src_scores[idx]),
            'harm': float(harms[idx]),
            'subset_gate_geomean': float(subset_g[idx]),
        })

    return {
        'pred': final_probs.argmax(axis=1),
        'probs': final_probs,
        'posterior': posterior,
        'candidate_rows': rows,
        'view_gate': view_gate,
        'view_evidence': view_evidence,
        'rule_prior': rule_prior,
        'posterior_entropy_norm': entropy_dist(posterior) / max(np.log(len(candidates)), 1e-12),
    }


def top2_gap(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return 1.0
    sorted_values = np.sort(values)
    return float(sorted_values[-1] - sorted_values[-2])


def agreement_gated_weights(weights_prob, weights_geo, fallback='prob', margin=0.1):
    if weights_prob.shape != weights_geo.shape:
        raise ValueError(f"weights_prob and weights_geo shapes differ: {weights_prob.shape} vs {weights_geo.shape}")

    num_views, num_classes = weights_prob.shape
    weights = np.zeros_like(weights_prob, dtype=np.float64)
    decisions = []
    for c in range(num_classes):
        top_prob = int(np.argmax(weights_prob[:, c]))
        top_geo = int(np.argmax(weights_geo[:, c]))
        sharp_prob = top2_gap(weights_prob[:, c])
        sharp_geo = top2_gap(weights_geo[:, c])

        if top_prob == top_geo:
            combined = weights_prob[:, c] * weights_geo[:, c]
            weights[:, c] = combined / max(combined.sum(), 1e-12)
            decisions.append('agree_product')
        elif fallback == 'uniform':
            weights[:, c] = np.ones(num_views, dtype=np.float64) / max(num_views, 1)
            decisions.append('conflict_uniform')
        elif fallback == 'sharper':
            if sharp_geo > sharp_prob + margin:
                weights[:, c] = weights_geo[:, c]
                decisions.append('conflict_geo_sharper')
            else:
                weights[:, c] = weights_prob[:, c]
                decisions.append('conflict_prob_sharper_or_default')
        elif fallback == 'average':
            weights[:, c] = 0.5 * (weights_prob[:, c] + weights_geo[:, c])
            weights[:, c] = weights[:, c] / max(weights[:, c].sum(), 1e-12)
            decisions.append('conflict_average')
        elif fallback == 'prob':
            weights[:, c] = weights_prob[:, c]
            decisions.append('conflict_prob_fallback')
        else:
            raise ValueError(f"Unsupported agreement gate fallback: {fallback}")
    return weights, np.array(decisions, dtype=object)


def corr_ignore_nan(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return None
    if np.std(a[mask]) < 1e-12 or np.std(b[mask]) < 1e-12:
        return None
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def safe_corr_zero(a, b):
    value = corr_ignore_nan(a, b)
    return 0.0 if value is None else value


def safe_auc(pos, neg):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    values = np.concatenate([pos, neg])
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64) + 1.0
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = np.mean(ranks[order[start:end]])
        start = end
    pos_ranks = ranks[:len(pos)]
    return float((np.sum(pos_ranks) - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def effect_size(pos, neg):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    pooled = np.sqrt((np.var(pos) + np.var(neg)) / 2.0)
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(pos) - np.mean(neg)) / pooled)


def rank_order_desc(values):
    return [int(idx) for idx in np.argsort(-np.asarray(values, dtype=np.float64))]


def rank_matches(a, b):
    return rank_order_desc(a) == rank_order_desc(b)


def top_name(view_names, values):
    return view_names[int(np.argmax(np.asarray(values, dtype=np.float64)))]


def js_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + 1e-12) / (b[mask] + 1e-12))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def prediction_distribution_penalty_global(source_labels, pred, num_classes):
    source_counts = np.bincount(source_labels, minlength=num_classes).astype(np.float64)
    pred_counts = np.bincount(pred, minlength=num_classes).astype(np.float64)
    source_prior = source_counts / max(source_counts.sum(), 1.0)
    pred_prior = pred_counts / max(pred_counts.sum(), 1.0)
    max_pred_frequency = float(pred_prior.max()) if len(pred_prior) else 0.0
    max_source_frequency = float(source_prior.max()) if len(source_prior) else 0.0
    entropy = float(-np.sum(pred_prior[pred_prior > 0] * np.log(pred_prior[pred_prior > 0] + 1e-12)))
    return {
        'js': js_divergence(source_prior, pred_prior),
        'collapse': max(0.0, max_pred_frequency - max_source_frequency),
        'entropy': entropy,
        'effective_classes': float(np.exp(entropy)),
        'max_pred_frequency': max_pred_frequency,
        'max_source_frequency': max_source_frequency,
    }


def compute_distance_signals(source_features, source_labels, query_features, query_pred, num_classes, var_eps):
    means, variances = fit_diagonal_gaussians(source_features, source_labels, num_classes, var_eps)
    distances = mahalanobis_distances(query_features, means, variances)
    row_idx = np.arange(len(query_pred))
    pred_dist = distances[row_idx, query_pred]
    margins = np.zeros(len(query_pred), dtype=np.float64)
    for i, pred_class in enumerate(query_pred):
        other = np.delete(distances[i], pred_class)
        margins[i] = float(np.min(other) - distances[i, pred_class]) if len(other) else 0.0
    return {
        'negative_distance': -pred_dist,
        'distance_to_pred': pred_dist,
        'geometry_margin': margins,
    }


def compute_softmax_signals(probs):
    probs = np.asarray(probs, dtype=np.float64)
    sorted_probs = np.sort(probs, axis=1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2] if probs.shape[1] > 1 else np.zeros_like(top1)
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    return {
        'confidence': top1,
        'softmax_margin': top1 - top2,
        'negative_entropy': -entropy,
    }


def build_audit_raw_signals(source_outputs, target_outputs, y_source, source_mask, views, num_classes, args):
    train_raw = {}
    target_raw = {}
    source_labels = y_source[source_mask]
    for view in views:
        train_pred = source_outputs[view]['pred'].astype(int)
        target_pred = target_outputs[view]['pred'].astype(int)
        train_signals = {}
        target_signals = {}
        train_signals.update(compute_softmax_signals(source_outputs[view]['probs']))
        target_signals.update(compute_softmax_signals(target_outputs[view]['probs']))
        train_signals.update(compute_distance_signals(
            source_outputs[view]['features'][source_mask],
            source_labels,
            source_outputs[view]['features'],
            train_pred,
            num_classes,
            args.var_eps,
        ))
        target_signals.update(compute_distance_signals(
            source_outputs[view]['features'][source_mask],
            source_labels,
            target_outputs[view]['features'],
            target_pred,
            num_classes,
            args.var_eps,
        ))
        train_raw[view] = train_signals
        target_raw[view] = target_signals
    return train_raw, target_raw


def source_anchor_components(train_raw, target_raw, views, audit_weights):
    components = {}
    for key, weight in audit_weights.items():
        if abs(float(weight)) < 1e-12:
            continue
        train_matrix = np.stack([train_raw[view][key] for view in views], axis=1).astype(np.float32)
        target_matrix = np.stack([target_raw[view][key] for view in views], axis=1).astype(np.float32)
        center = np.mean(train_matrix, axis=0, keepdims=True)
        scale = np.std(train_matrix, axis=0, keepdims=True)
        components[key] = {
            'weight': float(weight),
            'train_z': (train_matrix - center) / np.maximum(scale, 1e-6),
            'target_z': (target_matrix - center) / np.maximum(scale, 1e-6),
        }
    return components


def compute_source_anchor_health(components, train_penalties, target_penalties, views, distribution_weights):
    if components:
        first = next(iter(components.values()))
        train_health = np.zeros_like(first['train_z'], dtype=np.float32)
        target_health = np.zeros_like(first['target_z'], dtype=np.float32)
    else:
        train_health = np.zeros((0, len(views)), dtype=np.float32)
        target_health = np.zeros((0, len(views)), dtype=np.float32)

    for payload in components.values():
        train_health += payload['weight'] * payload['train_z']
        target_health += payload['weight'] * payload['target_z']

    train_h = np.mean(train_health, axis=0) if len(train_health) else np.zeros(len(views), dtype=np.float64)
    target_h = np.mean(target_health, axis=0) if len(target_health) else np.zeros(len(views), dtype=np.float64)
    train_h_median = np.median(train_health, axis=0) if len(train_health) else np.zeros(len(views), dtype=np.float64)
    target_h_median = np.median(target_health, axis=0) if len(target_health) else np.zeros(len(views), dtype=np.float64)
    for idx, view in enumerate(views):
        train_h[idx] -= distribution_weights['js'] * train_penalties[view]['js']
        train_h[idx] -= distribution_weights['collapse'] * train_penalties[view]['collapse']
        target_h[idx] -= distribution_weights['js'] * target_penalties[view]['js']
        target_h[idx] -= distribution_weights['collapse'] * target_penalties[view]['collapse']
        train_h_median[idx] -= distribution_weights['js'] * train_penalties[view]['js']
        train_h_median[idx] -= distribution_weights['collapse'] * train_penalties[view]['collapse']
        target_h_median[idx] -= distribution_weights['js'] * target_penalties[view]['js']
        target_h_median[idx] -= distribution_weights['collapse'] * target_penalties[view]['collapse']
    return train_h, target_h, train_h_median, target_h_median, train_health, target_health


def component_rows(views, components, distribution_weights, train_penalties, target_penalties):
    rows = []
    for view_idx, view in enumerate(views):
        train_signal_sum = 0.0
        target_signal_sum = 0.0
        for key, payload in sorted(components.items()):
            train_mean = float(np.mean(payload['train_z'][:, view_idx]))
            target_mean = float(np.mean(payload['target_z'][:, view_idx]))
            weighted_train = payload['weight'] * train_mean
            weighted_target = payload['weight'] * target_mean
            train_signal_sum += weighted_train
            target_signal_sum += weighted_target
            rows.append({
                'view': view,
                'component': key,
                'weight': float(payload['weight']),
                'train_component_mean': train_mean,
                'test_component_mean': target_mean,
                'train_weighted_component': float(weighted_train),
                'test_weighted_component': float(weighted_target),
                'shift_train_minus_test': float(weighted_train - weighted_target),
                'component_type': 'sample_signal',
            })
        for component, penalty_key in [('distribution_js_penalty', 'js'), ('collapse_penalty', 'collapse')]:
            weight = float(distribution_weights[penalty_key])
            train_value = -weight * float(train_penalties[view][penalty_key])
            target_value = -weight * float(target_penalties[view][penalty_key])
            rows.append({
                'view': view,
                'component': component,
                'weight': weight,
                'train_component_mean': float(train_penalties[view][penalty_key]),
                'test_component_mean': float(target_penalties[view][penalty_key]),
                'train_weighted_component': train_value,
                'test_weighted_component': target_value,
                'shift_train_minus_test': float(train_value - target_value),
                'component_type': 'distribution_penalty',
            })
        train_total = (
            train_signal_sum
            - distribution_weights['js'] * train_penalties[view]['js']
            - distribution_weights['collapse'] * train_penalties[view]['collapse']
        )
        target_total = (
            target_signal_sum
            - distribution_weights['js'] * target_penalties[view]['js']
            - distribution_weights['collapse'] * target_penalties[view]['collapse']
        )
        rows.append({
            'view': view,
            'component': 'total_source_anchor_H',
            'weight': 1.0,
            'train_component_mean': 0.0,
            'test_component_mean': 0.0,
            'train_weighted_component': float(train_total),
            'test_weighted_component': float(target_total),
            'shift_train_minus_test': float(train_total - target_total),
            'component_type': 'total',
        })
    return rows


def correct_wrong_rows(views, y_target, target_known_mask, target_preds, target_raw, target_health, components):
    rows = []
    known_idx = np.where(target_known_mask)[0]
    labels = y_target[known_idx]
    for view_idx, view in enumerate(views):
        pred = target_preds[view][known_idx]
        correct = pred == labels
        features = {
            'negative_distance': target_raw[view]['negative_distance'][known_idx],
            'D_pred': target_raw[view]['distance_to_pred'][known_idx],
            'geometry_margin': target_raw[view]['geometry_margin'][known_idx],
            'confidence': target_raw[view]['confidence'][known_idx],
            'softmax_margin': target_raw[view]['softmax_margin'][known_idx],
            'negative_entropy': target_raw[view]['negative_entropy'][known_idx],
            'source_anchored_sample_health': target_health[known_idx, view_idx] if len(target_health) else np.zeros(len(known_idx)),
        }
        for key, payload in sorted(components.items()):
            features[f'source_anchor_component_{key}'] = payload['weight'] * payload['target_z'][known_idx, view_idx]
        for feature, values in features.items():
            pos = np.asarray(values[correct], dtype=np.float64)
            neg = np.asarray(values[~correct], dtype=np.float64)
            rows.append({
                'view': view,
                'feature': feature,
                'n_correct': int(len(pos)),
                'n_wrong': int(len(neg)),
                'mean_correct': float(np.mean(pos)) if len(pos) else 0.0,
                'mean_wrong': float(np.mean(neg)) if len(neg) else 0.0,
                'mean_gap_correct_minus_wrong': float(np.mean(pos) - np.mean(neg)) if len(pos) and len(neg) else 0.0,
                'median_correct': float(np.median(pos)) if len(pos) else 0.0,
                'median_wrong': float(np.median(neg)) if len(neg) else 0.0,
                'auc_correct_vs_wrong': safe_auc(pos, neg),
                'effect_size_correct_vs_wrong': effect_size(pos, neg),
            })
    return rows


def expert_metrics_by_view(views, y_target, target_known_mask, target_preds):
    metrics = {'acc': [], 'macro_f1': [], 'weighted_f1': []}
    y_known = y_target[target_known_mask]
    for view in views:
        pred = target_preds[view][target_known_mask]
        metrics['acc'].append(safe_accuracy(y_target, target_preds[view], target_known_mask) or 0.0)
        _, _, macro_f1, _ = precision_recall_fscore_support(
            y_known, pred, average='macro', zero_division=0
        )
        _, _, weighted_f1, _ = precision_recall_fscore_support(
            y_known, pred, average='weighted', zero_division=0
        )
        metrics['macro_f1'].append(float(macro_f1))
        metrics['weighted_f1'].append(float(weighted_f1))
    return {key: np.asarray(value, dtype=np.float64) for key, value in metrics.items()}


def write_health_audit_report(path, rows, rank_summary):
    with open(path, 'w') as f:
        f.write('[Health Signal Audit]\n')
        f.write('If H rank does not match expert metrics, health fusion may amplify the wrong modality.\n\n')
        f.write('view | acc | macro_f1 | weighted_f1 | H_mean | H_median | W_mean | W_median | JS | collapse | source_train_H | source_test_H | source_shift\n')
        for row in rows:
            f.write(
                '{view} | {expert_acc:.6f} | {expert_macro_f1:.6f} | {expert_weighted_f1:.6f} | '
                '{H_mean:.6f} | {H_median:.6f} | {weight_from_H_mean_tau1:.6f} | '
                '{weight_from_H_median_tau1:.6f} | {js:.6f} | {collapse:.6f} | '
                '{source_anchor_train_H_mean:.6f} | {source_anchor_test_H_mean:.6f} | '
                '{source_anchor_shift_train_minus_test:.6f}\n'.format(**row)
            )
        f.write('\n[Rank Summary]\n')
        for key in sorted(rank_summary.keys()):
            f.write(f'{key}: {rank_summary[key]}\n')


def write_health_audit(
    output_dir, views, y_source, source_mask, y_target, target_known_mask,
    source_outputs, target_outputs, target_preds, num_classes,
    h_prob, h_geo, weights_prob, weights_geo, recall_by_view,
    target_supports, class_best_view, prob_selected, geo_selected, args
):
    audit_weights = {
        'geometry_margin': args.w_audit_geometry_margin,
        'negative_distance': args.w_audit_negative_distance,
        'softmax_margin': args.w_audit_softmax_margin,
        'negative_entropy': args.w_audit_negative_entropy,
        'confidence': args.w_audit_confidence,
    }
    distribution_weights = {
        'js': args.w_audit_distribution_shift,
        'collapse': args.w_audit_prediction_collapse,
    }
    train_raw, target_raw = build_audit_raw_signals(
        source_outputs, target_outputs, y_source, source_mask, views, num_classes, args
    )
    source_labels = y_source[source_mask]
    train_penalties = {
        view: prediction_distribution_penalty_global(source_labels, source_outputs[view]['pred'].astype(int), num_classes)
        for view in views
    }
    target_penalties = {
        view: prediction_distribution_penalty_global(source_labels, target_preds[view], num_classes)
        for view in views
    }
    components = source_anchor_components(train_raw, target_raw, views, audit_weights)
    train_h, target_h, train_h_median, target_h_median, train_health, target_health = compute_source_anchor_health(
        components, train_penalties, target_penalties, views, distribution_weights
    )
    weights_mean = softmax_np(target_h[None, :], axis=1)[0]
    weights_median = softmax_np(target_h_median[None, :], axis=1)[0]
    expert = expert_metrics_by_view(views, y_target, target_known_mask, target_preds)

    signal_rows = []
    for idx, view in enumerate(views):
        row = {
            'view': view,
            'expert_acc': float(expert['acc'][idx]),
            'expert_macro_f1': float(expert['macro_f1'][idx]),
            'expert_weighted_f1': float(expert['weighted_f1'][idx]),
            'd_pred_mean': float(np.mean(target_raw[view]['negative_distance'])),
            'd_pred_median': float(np.median(target_raw[view]['negative_distance'])),
            'd_pred_std': float(np.std(target_raw[view]['negative_distance'])),
            'margin_mean': float(np.mean(target_raw[view]['geometry_margin'])),
            'margin_median': float(np.median(target_raw[view]['geometry_margin'])),
            'margin_std': float(np.std(target_raw[view]['geometry_margin'])),
            'negative_distance_mean': float(np.mean(target_raw[view]['negative_distance'])),
            'negative_distance_median': float(np.median(target_raw[view]['negative_distance'])),
            'raw_health_mean': float(np.mean(target_health[:, idx])) if len(target_health) else 0.0,
            'raw_health_median': float(np.median(target_health[:, idx])) if len(target_health) else 0.0,
            'js': float(target_penalties[view]['js']),
            'collapse': float(target_penalties[view]['collapse']),
            'pred_entropy': float(target_penalties[view]['entropy']),
            'effective_pred_classes': float(target_penalties[view]['effective_classes']),
            'pred_dist_js_to_source': float(target_penalties[view]['js']),
            'max_pred_frequency': float(target_penalties[view]['max_pred_frequency']),
            'H_mean': float(target_h[idx]),
            'H_median': float(target_h_median[idx]),
            'weight_from_H_mean_tau1': float(weights_mean[idx]),
            'weight_from_H_median_tau1': float(weights_median[idx]),
            'source_anchor_train_H_mean': float(train_h[idx]),
            'source_anchor_test_H_mean': float(target_h[idx]),
            'source_anchor_shift_train_minus_test': float(train_h[idx] - target_h[idx]),
        }
        signal_rows.append(row)

    rank_summary = {
        'top_expert_view_acc': top_name(views, expert['acc']),
        'top_expert_view_macro_f1': top_name(views, expert['macro_f1']),
        'top_expert_view_weighted_f1': top_name(views, expert['weighted_f1']),
        'top_H_mean_view': top_name(views, target_h),
        'top_H_median_view': top_name(views, target_h_median),
        'H_mean_rank_matches_acc_rank': bool(rank_matches(target_h, expert['acc'])),
        'H_mean_rank_matches_macro_f1_rank': bool(rank_matches(target_h, expert['macro_f1'])),
        'H_mean_rank_matches_weighted_f1_rank': bool(rank_matches(target_h, expert['weighted_f1'])),
        'H_median_rank_matches_acc_rank': bool(rank_matches(target_h_median, expert['acc'])),
        'H_median_rank_matches_macro_f1_rank': bool(rank_matches(target_h_median, expert['macro_f1'])),
        'H_median_rank_matches_weighted_f1_rank': bool(rank_matches(target_h_median, expert['weighted_f1'])),
        'corr_H_mean_expert_acc': safe_corr_zero(target_h, expert['acc']),
        'corr_H_mean_expert_macro_f1': safe_corr_zero(target_h, expert['macro_f1']),
        'corr_H_mean_expert_weighted_f1': safe_corr_zero(target_h, expert['weighted_f1']),
        'corr_H_median_expert_acc': safe_corr_zero(target_h_median, expert['acc']),
        'corr_H_median_expert_macro_f1': safe_corr_zero(target_h_median, expert['macro_f1']),
        'corr_H_median_expert_weighted_f1': safe_corr_zero(target_h_median, expert['weighted_f1']),
        'health_gap_mean': float(np.sort(target_h)[-1] - np.sort(target_h)[-2]) if len(target_h) > 1 else 0.0,
        'health_gap_median': float(np.sort(target_h_median)[-1] - np.sort(target_h_median)[-2]) if len(target_h_median) > 1 else 0.0,
        'expert_acc_by_view': {view: float(expert['acc'][idx]) for idx, view in enumerate(views)},
        'expert_macro_f1_by_view': {view: float(expert['macro_f1'][idx]) for idx, view in enumerate(views)},
        'expert_weighted_f1_by_view': {view: float(expert['weighted_f1'][idx]) for idx, view in enumerate(views)},
        'H_mean_by_view': {view: float(target_h[idx]) for idx, view in enumerate(views)},
        'H_median_by_view': {view: float(target_h_median[idx]) for idx, view in enumerate(views)},
        'weight_from_H_mean_tau1_by_view': {view: float(weights_mean[idx]) for idx, view in enumerate(views)},
        'weight_from_H_median_tau1_by_view': {view: float(weights_median[idx]) for idx, view in enumerate(views)},
    }

    class_rows = []
    for c in range(num_classes):
        row = {
            'class_idx': c,
            'support': int(target_supports[c]),
            'best_recall_view': class_best_view[c],
            'prob_selected_view': prob_selected[c],
            'geo_selected_view': geo_selected[c],
        }
        for view in views:
            row[f'{view}_recall'] = recall_by_view[view][c]
        for idx, view in enumerate(views):
            row[f'H_prob_{view}'] = h_prob[idx, c]
            row[f'W_prob_{view}'] = weights_prob[idx, c]
            row[f'H_geo_{view}'] = h_geo[idx, c]
            row[f'W_geo_{view}'] = weights_geo[idx, c]
        class_rows.append(row)

    rank_summary.update({
        'class_prob_best_view_match_rate': float((prob_selected == class_best_view).mean()),
        'class_geo_best_view_match_rate': float((geo_selected == class_best_view).mean()),
    })

    correct_wrong = correct_wrong_rows(
        views, y_target, target_known_mask, target_preds, target_raw, target_health, components
    )
    components_summary = component_rows(views, components, distribution_weights, train_penalties, target_penalties)

    pd.DataFrame(signal_rows).to_csv(os.path.join(output_dir, 'health_audit_signal_summary.csv'), index=False)
    pd.DataFrame(components_summary).to_csv(os.path.join(output_dir, 'health_audit_component_summary.csv'), index=False)
    pd.DataFrame(correct_wrong).to_csv(os.path.join(output_dir, 'health_audit_correct_wrong_summary.csv'), index=False)
    pd.DataFrame(class_rows).to_csv(os.path.join(output_dir, 'health_audit_class_summary.csv'), index=False)
    with open(os.path.join(output_dir, 'health_audit_rank_summary.json'), 'w') as f:
        json.dump(rank_summary, f, indent=2, sort_keys=True)
    write_health_audit_report(os.path.join(output_dir, 'health_audit_signal_report.txt'), signal_rows, rank_summary)


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


def write_agreement_weights(path, weights, h_prob, h_geo, decisions, idx2label, views):
    rows = []
    for c, label in idx2label.items():
        row = {'class_idx': c, 'label': label}
        for i, view in enumerate(views):
            row[f'H_prob_{view}'] = h_prob[i, c]
            row[f'H_geo_{view}'] = h_geo[i, c]
            row[f'W_agreement_{view}'] = weights[i, c]
        row['selected_view'] = views[int(np.argmax(weights[:, c]))]
        row['agreement_gate_decision'] = decisions[c]
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
        'agreement_gated': 'agreement_gated_class_fusion',
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
    parser.add_argument('--agreement_gate_fallback', choices=['prob', 'uniform', 'sharper', 'average'], default='prob')
    parser.add_argument('--agreement_gate_margin', type=float, default=0.1)
    parser.add_argument('--audit_only', action='store_true')
    parser.add_argument('--write_health_audit', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--w_audit_geometry_margin', type=float, default=1.0)
    parser.add_argument('--w_audit_negative_distance', type=float, default=1.0)
    parser.add_argument('--w_audit_softmax_margin', type=float, default=0.0)
    parser.add_argument('--w_audit_negative_entropy', type=float, default=0.0)
    parser.add_argument('--w_audit_confidence', type=float, default=0.0)
    parser.add_argument('--w_audit_distribution_shift', type=float, default=1.0)
    parser.add_argument('--w_audit_prediction_collapse', type=float, default=1.0)
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
    source_val_path = os.path.join(args.source_dataset, 'val.csv.gz')
    if not os.path.exists(source_val_path):
        raise FileNotFoundError(f"HSPF source prior requires validation split: {source_val_path}")
    source_val_df = pd.read_csv(source_val_path, compression='gzip', index_col=0)
    target_df = pd.read_csv(args.target_csv, compression='gzip', index_col=0)
    if args.max_source_samples is not None:
        source_df = source_df.head(args.max_source_samples)
        source_val_df = source_val_df.head(args.max_source_samples)
    if args.max_target_samples is not None:
        target_df = target_df.head(args.max_target_samples)

    source_view_dfs = {view: preprocess_dataframe(source_df, view, label2idx) for view in views}
    source_val_view_dfs = {view: preprocess_dataframe(source_val_df, view, label2idx) for view in views}
    target_view_dfs = {view: preprocess_dataframe(target_df, view, label2idx) for view in views}

    source_len = len(source_view_dfs[views[0]])
    source_val_len = len(source_val_view_dfs[views[0]])
    target_len = len(target_view_dfs[views[0]])
    for view in views[1:]:
        if len(source_view_dfs[view]) != source_len:
            raise ValueError(f"Source sample mismatch after preprocessing: {views[0]}={source_len}, {view}={len(source_view_dfs[view])}")
        if len(source_val_view_dfs[view]) != source_val_len:
            raise ValueError(f"Source val sample mismatch after preprocessing: {views[0]}={source_val_len}, {view}={len(source_val_view_dfs[view])}")
        if len(target_view_dfs[view]) != target_len:
            raise ValueError(f"Target sample mismatch after preprocessing: {views[0]}={target_len}, {view}={len(target_view_dfs[view])}")

    source_mask = source_view_dfs[views[0]]['y_label'].notna().to_numpy()
    source_val_mask = source_val_view_dfs[views[0]]['y_label'].notna().to_numpy()
    target_known_mask = (
        target_view_dfs[views[0]]['y_label'].notna().to_numpy()
        if 'y_label' in target_view_dfs[views[0]].columns
        else np.zeros(target_len, dtype=bool)
    )
    y_source = source_view_dfs[views[0]]['y_label'].fillna(-1).astype(int).to_numpy()
    y_source_val = source_val_view_dfs[views[0]]['y_label'].fillna(-1).astype(int).to_numpy()
    y_target = target_view_dfs[views[0]]['y_label'].fillna(-1).astype(int).to_numpy()

    source_datasets = {view: make_dataset(source_view_dfs[view], view, has_label=True) for view in views}
    source_val_datasets = {view: make_dataset(source_val_view_dfs[view], view, has_label=True) for view in views}
    target_has_label = 'label' in target_view_dfs[views[0]].columns
    target_datasets = {view: make_dataset(target_view_dfs[view], view, has_label=target_has_label) for view in views}

    source_outputs, source_val_outputs, target_outputs = {}, {}, {}
    for view in views:
        source_cached = try_load_outputs_from_cache(
            'source', view, args.source_dataset, model_ts_by_view[view],
            num_classes, label2idx, source_datasets[view].num_rows, args
        )
        source_val_cached = try_load_outputs_from_cache(
            'source_val', view, source_val_path, model_ts_by_view[view],
            num_classes, label2idx, source_val_datasets[view].num_rows, args
        )
        target_cached = try_load_outputs_from_cache(
            'target', view, args.target_csv, model_ts_by_view[view],
            num_classes, label2idx, target_datasets[view].num_rows, args
        )
        if source_cached is not None and source_val_cached is not None and target_cached is not None:
            source_outputs[view] = source_cached
            source_val_outputs[view] = source_val_cached
            target_outputs[view] = target_cached
            continue
        if args.audit_only:
            missing = []
            if source_cached is None:
                missing.append(f'source_{view}.npz')
            if source_val_cached is None:
                missing.append(f'source_val_{view}.npz')
            if target_cached is None:
                missing.append(f'target_{view}.npz')
            cache_dir = args.cache_dir or os.path.join(args.output_dir, 'cache')
            raise FileNotFoundError(
                f"--audit_only requires existing matching cache files in {cache_dir}; missing or stale: {', '.join(missing)}"
            )

        model = load_view_model(model_paths[view], model_infos[view], view, num_classes)
        source_outputs[view] = source_cached or get_outputs_with_cache(
            model, source_datasets[view], 'source', view, args.source_dataset,
            model_ts_by_view[view], num_classes, label2idx, args
        )
        source_val_outputs[view] = source_val_cached or get_outputs_with_cache(
            model, source_val_datasets[view], 'source_val', view, source_val_path,
            model_ts_by_view[view], num_classes, label2idx, args
        )
        target_outputs[view] = target_cached or get_outputs_with_cache(
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

    weights_agreement, agreement_decisions = agreement_gated_weights(
        weights_prob,
        weights_geo,
        fallback=args.agreement_gate_fallback,
        margin=args.agreement_gate_margin,
    )
    probs_agreement = fuse_probs(weights_agreement, target_probs, views)
    pred_agreement = probs_agreement.argmax(axis=1)

    source_val_probs = {view: source_val_outputs[view]['probs'] for view in views}
    hspf_candidates = hspf_build_candidates(
        views,
        target_probs,
        weights_prob,
        weights_geo,
        weights_agreement,
    )
    hspf = hspf_run(
        hspf_candidates,
        views,
        target_probs,
        source_outputs,
        target_outputs,
        src_prior,
        source_val_probs,
        y_source_val,
        source_val_mask,
        weights_prob,
        weights_geo,
        weights_agreement,
        num_classes,
    )
    pred_hspf = hspf['pred']

    prob_selected = np.array([views[i] for i in weights_prob.argmax(axis=0)], dtype=object)
    geo_selected = np.array([views[i] for i in weights_geo.argmax(axis=0)], dtype=object)
    agreement_selected = np.array([views[i] for i in weights_agreement.argmax(axis=0)], dtype=object)
    health_best_view_match_rate_prob = float((prob_selected == class_best_view).mean())
    health_best_view_match_rate_geo = float((geo_selected == class_best_view).mean())
    health_best_view_match_rate_agreement = float((agreement_selected == class_best_view).mean())

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
        for i, view in enumerate(views):
            row[f'W_agreement_{view}'] = weights_agreement[i, c]
        row['agreement_selected_view'] = agreement_selected[c]
        row['agreement_gate_decision'] = agreement_decisions[c]
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
        'agreement_gate_fallback': args.agreement_gate_fallback,
        'agreement_gate_margin': args.agreement_gate_margin,
        'num_classes': num_classes,
        'num_source_samples': int(source_len),
        'num_source_val_samples': int(source_val_len),
        'num_target_samples': int(target_len),
        'num_target_known_label_samples': int(target_known_mask.sum()),
        **acc_by_view,
        'best_single_acc': best_single_acc,
        'avg_prob_acc': safe_accuracy(y_target, pred_avg, target_known_mask),
        'prob_only_class_fusion_acc': safe_accuracy(y_target, pred_prob, target_known_mask),
        'geometry_class_fusion_acc': safe_accuracy(y_target, pred_geo, target_known_mask),
        'agreement_gated_class_fusion_acc': safe_accuracy(y_target, pred_agreement, target_known_mask),
        'hspf_acc_diagnostic': safe_accuracy(y_target, pred_hspf, target_known_mask),
        'hspf_top_candidate': hspf['candidate_rows'][0]['candidate'] if hspf['candidate_rows'] else None,
        'hspf_top_posterior': hspf['candidate_rows'][0]['posterior'] if hspf['candidate_rows'] else None,
        'hspf_entropy_norm': hspf['posterior_entropy_norm'],
        'hspf_view_gate': hspf['view_gate'],
        'hspf_rule_prior': hspf['rule_prior'],
        'class_oracle_acc': safe_accuracy(y_target, y_class_oracle, target_known_mask),
        'sample_oracle_acc': safe_accuracy(y_target, y_sample_oracle, target_known_mask),
        'health_best_view_match_rate_prob': health_best_view_match_rate_prob,
        'health_best_view_match_rate_geo': health_best_view_match_rate_geo,
        'health_best_view_match_rate_agreement': health_best_view_match_rate_agreement,
        'agreement_gate_decision_counts': {
            str(name): int((agreement_decisions == name).sum()) for name in sorted(set(agreement_decisions))
        },
        'corr_h_prob_selected_with_best_recall': corr_ignore_nan(h_prob_best, best_recall),
        'corr_h_geo_selected_with_best_recall': corr_ignore_nan(h_geo_best, best_recall),
        'warnings': [],
    }
    if summary['sample_oracle_acc'] is not None and summary['sample_oracle_acc'] + 1e-12 < summary['best_single_acc']:
        summary['warnings'].append('sample_oracle_acc is lower than best_single_acc; check labels/predictions.')
    if summary['class_oracle_acc'] is not None and summary['class_oracle_acc'] + 1e-12 < summary['best_single_acc']:
        summary['warnings'].append('class_oracle_acc is lower than best_single_acc; this may happen if target has missing classes.')
    if target_known_mask.sum() > 0:
        _, _, hspf_weighted_f1, _ = precision_recall_fscore_support(
            y_target[target_known_mask], pred_hspf[target_known_mask], average='weighted', zero_division=0
        )
        _, _, hspf_macro_f1, _ = precision_recall_fscore_support(
            y_target[target_known_mask], pred_hspf[target_known_mask], average='macro', zero_division=0
        )
        summary['hspf_weighted_f1'] = float(hspf_weighted_f1)
        summary['hspf_macro_f1_diagnostic'] = float(hspf_macro_f1)
        summary['primary_portfolio_weighted_f1'] = float(hspf_weighted_f1)
        summary['primary_portfolio_acc_diagnostic'] = summary['hspf_acc_diagnostic']
        summary['primary_portfolio_macro_f1_diagnostic'] = float(hspf_macro_f1)
    else:
        summary['hspf_weighted_f1'] = None
        summary['hspf_macro_f1_diagnostic'] = None
        summary['primary_portfolio_weighted_f1'] = None
        summary['primary_portfolio_acc_diagnostic'] = None
        summary['primary_portfolio_macro_f1_diagnostic'] = None

    target_out = target_view_dfs[views[0]].copy()
    target_out['y_true_idx'] = y_target
    for view in views:
        target_out[f'pred_{view}_idx'] = target_preds[view]
    target_out['pred_avg_idx'] = pred_avg
    target_out['pred_prob_only_idx'] = pred_prob
    target_out['pred_geometry_idx'] = pred_geo
    target_out['pred_agreement_gated_idx'] = pred_agreement
    target_out['pred_hspf_idx'] = pred_hspf
    target_out['pred_class_oracle_idx'] = y_class_oracle
    target_out['pred_sample_oracle_idx'] = y_sample_oracle
    for view in views:
        target_out[f'pred_{view}'] = [idx2label[i] for i in target_preds[view]]
    target_out['pred_avg'] = [idx2label[i] for i in pred_avg]
    target_out['pred_prob_only'] = [idx2label[i] for i in pred_prob]
    target_out['pred_geometry'] = [idx2label[i] for i in pred_geo]
    target_out['pred_agreement_gated'] = [idx2label[i] for i in pred_agreement]
    target_out['pred_hspf'] = [idx2label[i] for i in pred_hspf]
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
        'agreement_gated_class_fusion': pred_agreement,
        'hspf': pred_hspf,
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
    pd.DataFrame(hspf['candidate_rows']).to_csv(os.path.join(args.output_dir, 'hspf_candidate_ranking.csv'), index=False)
    hspf_view_rows = []
    for view in views:
        row = {'view': view, 'gate': hspf['view_gate'][view]}
        row.update(hspf['view_evidence'][view])
        hspf_view_rows.append(row)
    pd.DataFrame(hspf_view_rows).to_csv(os.path.join(args.output_dir, 'hspf_view_gates.csv'), index=False)
    pd.DataFrame([
        {'rule': rule, 'prior': prior}
        for rule, prior in hspf['rule_prior'].items()
    ]).to_csv(os.path.join(args.output_dir, 'hspf_rule_priors.csv'), index=False)
    write_weights(os.path.join(args.output_dir, 'class_weights_prob.csv'), weights_prob, h_prob, idx2label, views)
    write_weights(os.path.join(args.output_dir, 'class_weights_geometry.csv'), weights_geo, h_geo, idx2label, views)
    write_agreement_weights(
        os.path.join(args.output_dir, 'class_weights_agreement_gated.csv'),
        weights_agreement, h_prob, h_geo, agreement_decisions, idx2label, views
    )
    write_classification_reports(args.output_dir, report_texts, report_dicts, report_df)
    if args.write_health_audit:
        write_health_audit(
            args.output_dir, views, y_source, source_mask, y_target, target_known_mask,
            source_outputs, target_outputs, target_preds, num_classes,
            h_prob, h_geo, weights_prob, weights_geo, recall_by_view,
            target_supports, class_best_view, prob_selected, geo_selected, args
        )

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
