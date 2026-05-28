import argparse
import copy
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from tqdm import tqdm
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast


device = "cuda" if torch.cuda.is_available() else "cpu"

max_length_ps = 256
max_length_bytes = 512
model_name = "MM4flow"
premodel_name_ps = "BERT-ps"
premodel_name_raw = "BERT-bytes"

tokenizer_ps = BertTokenizerFast.from_pretrained("tokenizer_bert/ps_tokenizer")
tokenizer_bytes = BertTokenizerFast.from_pretrained("tokenizer_bert/bytes_tokenizer")
drop_empty = lambda x: x if x != "(empty)" else np.nan


class UniModalClassifier(nn.Module):
    def __init__(self, config, num_classes, modality):
        super().__init__()
        if modality not in ["ps", "byte"]:
            raise ValueError(f"Unsupported modality: {modality}")
        self.modality = modality
        self.encoder = BertForMaskedLM(config)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, inputs, return_features=False):
        if self.modality == "ps":
            outputs = self.encoder.bert(input_ids=inputs["ps"], attention_mask=inputs["ps_attention_mask"])
        else:
            outputs = self.encoder.bert(
                input_ids=inputs["raw"],
                attention_mask=inputs["raw_attention_mask"],
                token_type_ids=inputs["raw_token_type_ids"],
            )
        features = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(features)
        if return_features:
            return {"logits": logits, "features": features}
        return {"logits": logits}


class MMClassifier(nn.Module):
    def __init__(self, ps_config, bytes_config, num_classes):
        super().__init__()
        self.ps_encoder = BertForMaskedLM(ps_config)
        self.bytes_encoder = BertForMaskedLM(bytes_config)
        self.ps_cross_attention = nn.MultiheadAttention(embed_dim=ps_config.hidden_size, num_heads=4, batch_first=True)
        self.bytes_cross_attention = nn.MultiheadAttention(
            embed_dim=bytes_config.hidden_size, num_heads=4, batch_first=True
        )
        self.classifier = nn.Sequential(nn.Linear(ps_config.hidden_size + bytes_config.hidden_size, num_classes))

    def forward(self, inputs, return_features=False):
        ps_outputs = self.ps_encoder.bert(input_ids=inputs["ps"], attention_mask=inputs["ps_attention_mask"])
        raw_outputs = self.bytes_encoder.bert(
            input_ids=inputs["raw"],
            attention_mask=inputs["raw_attention_mask"],
            token_type_ids=inputs["raw_token_type_ids"],
        )

        outputs = torch.concat([ps_outputs.last_hidden_state, raw_outputs.last_hidden_state], dim=1)
        key_padding_mask = (1 - torch.concat([inputs["ps_attention_mask"], inputs["raw_attention_mask"]], dim=1)).bool()
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
            return {"logits": logits, "features": features}
        return {"logits": logits}


def parse_views(views_arg):
    views = [v.strip() for v in views_arg.split(",") if v.strip()]
    allowed = {"ps", "byte", "mm"}
    if not views:
        raise ValueError("--views must contain at least one view.")
    unsupported = [v for v in views if v not in allowed]
    if unsupported:
        raise ValueError(f"Unsupported views: {unsupported}. Supported views: {sorted(allowed)}")
    return views


def func_ps(ps):
    r = []
    for burst in str(ps).split(","):
        if not burst or burst == "(empty)":
            continue
        p_len, p_count = burst.split(":")
        r += [f"p{p_len}t"] * int(p_count)
        if len(r) > max_length_ps:
            break
    return " ".join(r[0:max_length_ps])


def func_bytes(s):
    s = "" if pd.isna(s) else str(s)
    return " ".join([s[i : i + 2] for i in range(0, len(s), 2)])


def preprocess_dataframe(df, view, label2idx=None):
    df = df.copy()
    if label2idx is not None and "label" in df.columns:
        df["y_label"] = df["label"].map(label2idx)
    if view in ["ps", "mm"] and "ps" in df.columns:
        df["ps"] = df["ps"].apply(func_ps)
    if view in ["byte", "mm"] and "fwd_raw" in df.columns and "bwd_raw" in df.columns:
        df["fwd_raw"] = df["fwd_raw"].apply(drop_empty).fillna(" ").apply(func_bytes)
        df["bwd_raw"] = df["bwd_raw"].apply(drop_empty).fillna(" ").apply(func_bytes)
    return df


def dataset_columns(view, has_label):
    if view == "ps":
        return ["ps"] + (["y_label"] if has_label else [])
    if view == "byte":
        return ["fwd_raw", "bwd_raw"] + (["y_label"] if has_label else [])
    return ["fwd_raw", "bwd_raw", "ps"] + (["y_label"] if has_label else [])


def tensor_columns(view, has_label):
    cols = []
    if view in ["ps", "mm"]:
        cols += ["ps", "ps_attention_mask"]
    if view in ["byte", "mm"]:
        cols += ["raw", "raw_attention_mask", "raw_token_type_ids"]
    if has_label:
        cols += ["y_label"]
    return cols


def make_encoder(view):
    def encode(examples):
        encoded = {}
        if view in ["ps", "mm"]:
            ps = tokenizer_ps(
                examples["ps"],
                truncation=True,
                padding="max_length",
                max_length=max_length_ps,
                return_special_tokens_mask=True,
            )
            encoded.update({"ps": ps["input_ids"], "ps_attention_mask": ps["attention_mask"]})
        if view in ["byte", "mm"]:
            raw = tokenizer_bytes(
                list(zip(examples["fwd_raw"], examples["bwd_raw"])),
                truncation=True,
                padding="max_length",
                max_length=max_length_bytes,
                return_special_tokens_mask=True,
            )
            encoded.update(
                {"raw": raw["input_ids"], "raw_attention_mask": raw["attention_mask"], "raw_token_type_ids": raw["token_type_ids"]}
            )
        return encoded

    return encode


def make_dataset(df, view, has_label):
    dataset = Dataset.from_pandas(df[dataset_columns(view, has_label)])
    dataset = dataset.map(make_encoder(view), batched=True)
    remove_columns = [c for c in ["fwd_raw", "bwd_raw", "uid", "__index_level_0__"] if c in dataset.column_names]
    if remove_columns:
        dataset = dataset.remove_columns(remove_columns)
    dataset.set_format(type="torch", columns=tensor_columns(view, has_label))
    return dataset


def resolve_model_dirs(model_ts):
    model_root = os.path.join("model-classifier", model_ts, model_name)
    info_path = os.path.join(model_root, "info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    with open(info_path, "r") as f:
        info = json.load(f)
    return model_root, info


def load_configs_from_info(info, view):
    ps_config, bytes_config = None, None
    if view in ["ps", "mm"]:
        ps_hp_path = os.path.join("model", f"{premodel_name_ps}_{info['pre_timestamp_ps']}", "hyperparameters.json")
        with open(ps_hp_path, "r") as f:
            hp = json.load(f)
        ps_config = BertConfig(
            vocab_size=len(tokenizer_ps.get_vocab()),
            max_position_embeddings=max_length_ps,
            hidden_size=hp["d_model"],
            num_hidden_layers=hp["n_layer"],
            num_attention_heads=hp["n_head"],
            intermediate_size=hp["dim_ff"],
        )
    if view in ["byte", "mm"]:
        raw_hp_path = os.path.join("model", f"{premodel_name_raw}_{info['pre_timestamp_raw']}", "hyperparameters.json")
        with open(raw_hp_path, "r") as f:
            hp = json.load(f)
        bytes_config = BertConfig(
            vocab_size=len(tokenizer_bytes.get_vocab()),
            max_position_embeddings=max_length_bytes,
            hidden_size=hp["d_model"],
            num_hidden_layers=hp["n_layer"],
            num_attention_heads=hp["n_head"],
            intermediate_size=hp["dim_ff"],
        )
    return ps_config, bytes_config


def build_model_for_view(view, info):
    label2idx = info["label2idx"]
    num_classes = len(label2idx)
    ps_config, bytes_config = load_configs_from_info(info, view)
    if view == "mm":
        return MMClassifier(ps_config, bytes_config, num_classes=num_classes)
    if view == "ps":
        return UniModalClassifier(ps_config, num_classes=num_classes, modality="ps")
    return UniModalClassifier(bytes_config, num_classes=num_classes, modality="byte")


def model_dir_for_view(model_root, view):
    model_type = f"{view}-finetune"
    return os.path.join(model_root, model_type)


def classifier_module(model, view):
    if view == "mm":
        return model.classifier[0]
    return model.classifier


def load_base_model(view, model_ts):
    model_root, info = resolve_model_dirs(model_ts)
    model_dir = model_dir_for_view(model_root, view)
    state_path = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Missing base pytorch_model.bin: {state_path}")
    model = build_model_for_view(view, info)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model, model_root, model_dir, info


def extract_features(model, dataset, infer_batch_size, has_label):
    model = model.to(device)
    model.eval()
    dl = torch.utils.data.DataLoader(dataset, batch_size=infer_batch_size, shuffle=False)
    feats, labels = [], []
    with torch.no_grad():
        for batch in tqdm(dl, desc="extract", unit="batch"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch, return_features=True)
            feats.append(out["features"].detach().cpu())
            if has_label:
                labels.append(batch["y_label"].detach().cpu())
    feats = torch.cat(feats, dim=0)
    if has_label:
        labels = torch.cat(labels, dim=0)
        return feats, labels
    return feats, None


def try_load_reuse_features(reuse_dir, split_name, view):
    if reuse_dir is None:
        return None
    path = os.path.join(reuse_dir, f"{split_name}_{view}.npz")
    if not os.path.exists(path):
        return None
    payload = np.load(path)
    if "features" not in payload:
        return None
    feats = torch.from_numpy(np.asarray(payload["features"], dtype=np.float32))
    labels = None
    if "labels" in payload:
        labels = torch.from_numpy(np.asarray(payload["labels"], dtype=np.int64))
    return feats, labels


def maybe_save_features(save_dir, split_name, view, features, labels):
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{split_name}_{view}.npz")
    payload = {"features": np.asarray(features, dtype=np.float32)}
    if labels is not None:
        payload["labels"] = np.asarray(labels, dtype=np.int64)
    np.savez(path, **payload)


def compute_class_variances(features, labels, num_classes, variance_floor):
    features_np = features.numpy()
    labels_np = labels.numpy()
    global_var = np.var(features_np, axis=0, ddof=1) if features_np.shape[0] > 1 else np.var(features_np, axis=0)
    sigma2 = np.zeros((num_classes, features_np.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        idx = np.where(labels_np == c)[0]
        if len(idx) < 2:
            sigma2[c] = global_var
        else:
            sigma2[c] = np.var(features_np[idx], axis=0, ddof=1)
    sigma2 = np.maximum(sigma2, float(variance_floor)).astype(np.float32)
    return torch.from_numpy(sigma2)


def weighted_f1_from_preds(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return float(f1), float(p), float(r)


def macro_f1_from_preds(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return float(f1), float(p), float(r)


def evaluate_head_preds(head, features, labels=None, report_topk=3):
    head.eval()
    with torch.no_grad():
        logits = head(features.to(device))
        probs = torch.softmax(logits, dim=1).detach().cpu()
        preds = torch.argmax(probs, dim=1).numpy()
    out = {"preds": preds, "probs": probs.numpy()}
    if labels is not None:
        y_true = labels.detach().cpu().numpy()
        acc = float(accuracy_score(y_true, preds))
        wf1, wp, wr = weighted_f1_from_preds(y_true, preds)
        mf1, mp, mr = macro_f1_from_preds(y_true, preds)
        out.update(
            {
                "acc": acc,
                "weighted_f1": wf1,
                "macro_f1": mf1,
                "weighted_precision": wp,
                "weighted_recall": wr,
                "macro_precision": mp,
                "macro_recall": mr,
            }
        )
        if report_topk and report_topk >= 2:
            topk = min(int(report_topk), probs.shape[1])
            top_idx = np.argsort(-out["probs"], axis=1)[:, :topk]
            y_true_col = y_true.reshape(-1, 1)
            for k in [2, 3]:
                if k <= topk:
                    out[f"top{k}"] = float(np.mean(np.any(top_idx[:, :k] == y_true_col, axis=1)))
    return out


def probs_entropy(probs):
    p = np.asarray(probs, dtype=np.float64)
    return -np.sum(p * np.log(p + 1e-12), axis=1)


def pred_distribution(preds, num_classes):
    preds = np.asarray(preds, dtype=np.int64)
    counts = np.bincount(preds, minlength=num_classes).astype(np.int64)
    fracs = counts.astype(np.float64) / max(1, preds.shape[0])
    return counts, fracs


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def compute_delta(mu_s, mu_t, z_s, delta_normalize):
    delta_raw = mu_t - mu_s
    raw_norm = float(torch.norm(delta_raw).item())
    if not delta_normalize or raw_norm < 1e-12:
        return delta_raw, raw_norm, float(torch.norm(delta_raw).item())
    target_norm = float(torch.norm(z_s, dim=1).mean().item())
    delta = delta_raw / (raw_norm + 1e-12) * target_norm
    return delta, raw_norm, float(torch.norm(delta).item())


def robust_component_eval(head, head_orig_state, z, y, sigma2_by_class, delta, shift_rho, adv_radius, lambda1, lambda2, variance_floor):
    head.eval()
    z = z.to(device)
    y = y.to(device)
    sigma2_by_class = sigma2_by_class.to(device)
    delta = delta.to(device)

    std_alpha = float(np.sqrt(lambda1)) if lambda1 > 0 else 0.0
    std_beta = float(np.sqrt(lambda2)) if lambda2 > 0 else 0.0

    with torch.no_grad():
        logits = head(z)
        clean_ce = float(nn.functional.cross_entropy(logits, y).item())

    # rand
    with torch.no_grad():
        alpha = torch.normal(mean=torch.ones_like(z), std=std_alpha) if std_alpha > 0 else torch.ones_like(z)
        beta = torch.normal(mean=torch.zeros_like(z), std=std_beta) if std_beta > 0 else torch.zeros_like(z)
        sigma2 = sigma2_by_class[y].clamp_min(float(variance_floor))
        gamma = torch.normal(mean=torch.zeros_like(z), std=torch.sqrt(sigma2))
        z_rand = alpha * z + beta + gamma
        aug_ce = float(nn.functional.cross_entropy(head(z_rand), y).item())
        z_shift = z_rand + float(shift_rho) * delta
        shift_ce = float(nn.functional.cross_entropy(head(z_shift), y).item())

    # adv
    z_req = z.detach().requires_grad_(True)
    logits_adv = head(z_req)
    ce_adv_clean = nn.functional.cross_entropy(logits_adv, y)
    grad = torch.autograd.grad(ce_adv_clean, z_req, create_graph=False)[0]
    grad_norm = torch.norm(grad, dim=1, keepdim=True)
    eps = float(adv_radius) * grad / (grad_norm + 1e-12)
    z_adv = (z + eps).detach()
    with torch.no_grad():
        adv_ce = float(nn.functional.cross_entropy(head(z_adv), y).item())

    # anchor
    anchor = 0.0
    if head_orig_state is not None:
        for name, param in head.named_parameters():
            anchor += float(torch.sum((param.detach().cpu() - head_orig_state[name]) ** 2).item())
    return {
        "clean_ce": clean_ce,
        "aug_ce": aug_ce,
        "shift_ce": shift_ce,
        "adv_ce": adv_ce,
        "anchor": anchor,
    }


def train_shift_robust_head(
    head,
    train_features,
    train_labels,
    val_features,
    val_labels,
    sigma2_by_class,
    delta,
    *,
    steps,
    lr,
    lambda1,
    lambda2,
    shift_rho,
    adv_radius,
    w_aug,
    w_shift,
    w_adv,
    w_anchor,
    variance_floor,
    batch_size,
    eval_every,
    select_best,
    component_eval_subset,
    report_topk,
):
    head = head.to(device)
    head.train()
    head_orig_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in head.state_dict().items()})

    for p in head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(head.parameters(), lr=lr, eps=1e-8)

    sigma2_by_class = sigma2_by_class.to(device)
    delta = delta.to(device)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    std_alpha = float(np.sqrt(lambda1)) if lambda1 > 0 else 0.0
    std_beta = float(np.sqrt(lambda2)) if lambda2 > 0 else 0.0

    rng = np.random.default_rng(42)
    n = train_features.shape[0]

    base_val = evaluate_head_preds(head, val_features, val_labels, report_topk=report_topk)
    base_val_weighted_f1 = float(base_val.get("weighted_f1", 0.0))

    best = {"step": -1, "metrics": None, "state": None, "components": None}

    subset_idx = None
    if component_eval_subset is not None and component_eval_subset > 0:
        m = min(int(component_eval_subset), n)
        subset_idx = np.arange(m, dtype=np.int64)

    for step in range(1, steps + 1):
        idx = rng.integers(low=0, high=n, size=batch_size, endpoint=False)
        z = train_features[idx].detach()
        y = train_labels[idx]

        alpha = torch.normal(mean=torch.ones_like(z), std=std_alpha) if std_alpha > 0 else torch.ones_like(z)
        beta = torch.normal(mean=torch.zeros_like(z), std=std_beta) if std_beta > 0 else torch.zeros_like(z)
        sigma2 = sigma2_by_class[y].clamp_min(float(variance_floor))
        gamma = torch.normal(mean=torch.zeros_like(z), std=torch.sqrt(sigma2))
        z_rand = alpha * z + beta + gamma
        z_shift = z_rand + float(shift_rho) * delta

        z_req = z.detach().requires_grad_(True)
        logits_clean_for_adv = head(z_req)
        clean_ce_for_adv = nn.functional.cross_entropy(logits_clean_for_adv, y)
        grad = torch.autograd.grad(clean_ce_for_adv, z_req, create_graph=False)[0]
        grad_norm = torch.norm(grad, dim=1, keepdim=True)
        eps = float(adv_radius) * grad / (grad_norm + 1e-12)
        z_adv = (z + eps).detach()

        logits_clean = head(z)
        logits_rand = head(z_rand)
        logits_shift = head(z_shift)
        logits_adv = head(z_adv)

        clean_ce = nn.functional.cross_entropy(logits_clean, y)
        aug_ce = nn.functional.cross_entropy(logits_rand, y)
        shift_ce = nn.functional.cross_entropy(logits_shift, y)
        adv_ce = nn.functional.cross_entropy(logits_adv, y)

        anchor = 0.0
        if w_anchor > 0:
            for name, param in head.named_parameters():
                anchor = anchor + torch.sum((param - head_orig_state[name].to(device)) ** 2)

        loss = clean_ce + float(w_aug) * aug_ce + float(w_shift) * shift_ce + float(w_adv) * adv_ce + float(w_anchor) * anchor

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if eval_every > 0 and (step % eval_every == 0 or step == steps):
            metrics = evaluate_head_preds(head, val_features, val_labels, report_topk=report_topk)
            key = select_best
            score = float(metrics.get(key, metrics.get("weighted_f1", 0.0)))
            if best["metrics"] is None or score > float(best["metrics"].get(key, -1e9)):
                best["step"] = step
                best["metrics"] = metrics
                best["state"] = copy.deepcopy({k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
                if subset_idx is not None:
                    z_sub = train_features[subset_idx].detach().cpu()
                    y_sub = train_labels[subset_idx].detach().cpu()
                    comps = robust_component_eval(
                        head,
                        head_orig_state,
                        z_sub,
                        y_sub,
                        sigma2_by_class.detach().cpu(),
                        delta.detach().cpu(),
                        shift_rho,
                        adv_radius,
                        lambda1,
                        lambda2,
                        variance_floor,
                    )
                    best["components"] = comps

    if best["state"] is not None:
        head.load_state_dict(best["state"])

    return base_val_weighted_f1, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", required=True)
    parser.add_argument("--target_csv", required=True)
    parser.add_argument("--views", default="ps,byte,mm")
    parser.add_argument("--ps_model_ts", default=None)
    parser.add_argument("--byte_model_ts", default=None)
    parser.add_argument("--mm_model_ts", default=None)
    parser.add_argument("--output_suffix", default="_shift_robust_cls")
    parser.add_argument("--reuse_feature_cache_dir", default=None)
    parser.add_argument("--save_feature_cache_dir", default=None)
    parser.add_argument("--output_dir", default=None)

    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda1", type=float, default=0.01)
    parser.add_argument("--lambda2", type=float, default=0.01)
    parser.add_argument("--shift_rho", type=float, default=0.5)
    parser.add_argument("--adv_radius", type=float, default=0.05)
    parser.add_argument("--w_aug", type=float, default=1.0)
    parser.add_argument("--w_shift", type=float, default=1.0)
    parser.add_argument("--w_adv", type=float, default=0.5)
    parser.add_argument("--w_anchor", type=float, default=1e-3)
    parser.add_argument("--variance_floor", type=float, default=1e-8)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--select_best", choices=["weighted_f1", "macro_f1", "acc"], default="weighted_f1")

    parser.add_argument("--delta_normalize", action="store_true")
    parser.add_argument("--report_topk", type=int, default=3)
    parser.add_argument("--eval_target_after_train", action="store_true")
    parser.add_argument("--save_target_pred_csv", action="store_true")

    parser.add_argument("--component_eval_subset", type=int, default=2048)
    parser.add_argument("--max_train_features", type=int, default=None)
    parser.add_argument("--max_val_features", type=int, default=None)
    parser.add_argument("--max_target_features", type=int, default=None)
    args = parser.parse_args()

    views = parse_views(args.views)
    model_ts_by_view = {"ps": args.ps_model_ts, "byte": args.byte_model_ts, "mm": args.mm_model_ts}
    for v in views:
        if model_ts_by_view.get(v) is None:
            raise ValueError(f"Missing --{v}_model_ts for view={v}")

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        args.output_dir = os.path.join("outputs", "mm4flow", f"shift_robust_{stamp}")
    ensure_dir(args.output_dir)

    base_view = "ps" if "ps" in views else views[0]
    _, _, _, base_info = load_base_model(base_view, model_ts_by_view[base_view])
    label2idx = base_info["label2idx"]
    idx2label = {int(v): k for k, v in label2idx.items()}
    num_classes = len(label2idx)

    train_csv = os.path.join(args.source_dataset, "train.csv.gz")
    val_csv = os.path.join(args.source_dataset, "val.csv.gz")
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError(f"Missing train/val csv under {args.source_dataset}")
    if not os.path.exists(args.target_csv):
        raise FileNotFoundError(f"Missing target_csv: {args.target_csv}")

    df_train_all = pd.read_csv(train_csv, compression="gzip", index_col=0)
    df_val_all = pd.read_csv(val_csv, compression="gzip", index_col=0)
    df_target_all = pd.read_csv(args.target_csv, compression="gzip", index_col=0)

    target_has_label = "label" in df_target_all.columns
    target_labels = None
    if target_has_label:
        df_target_all = df_target_all[df_target_all["label"].isin(label2idx.keys())].copy()
        if len(df_target_all) > 0:
            df_target_all["y_label"] = df_target_all["label"].map(label2idx)
            target_labels = torch.from_numpy(df_target_all["y_label"].astype(np.int64).values)
        else:
            target_has_label = False
            target_labels = None

    summary = {
        "source_dataset": args.source_dataset,
        "target_csv": args.target_csv,
        "views": views,
        "device": device,
        "steps": args.steps,
        "lr": args.lr,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "shift_rho": args.shift_rho,
        "adv_radius": args.adv_radius,
        "w_aug": args.w_aug,
        "w_shift": args.w_shift,
        "w_adv": args.w_adv,
        "w_anchor": args.w_anchor,
        "variance_floor": args.variance_floor,
        "batch_size": args.batch_size,
        "infer_batch_size": args.infer_batch_size,
        "eval_every": args.eval_every,
        "select_best": args.select_best,
        "output_suffix": args.output_suffix,
        "reuse_feature_cache_dir": args.reuse_feature_cache_dir,
        "save_feature_cache_dir": args.save_feature_cache_dir,
        "output_dir": args.output_dir,
        "delta_normalize": bool(args.delta_normalize),
        "report_topk": int(args.report_topk),
        "eval_target_after_train": bool(args.eval_target_after_train),
        "save_target_pred_csv": bool(args.save_target_pred_csv),
    }

    for view in views:
        model_ts = model_ts_by_view[view]
        model, model_root, model_dir, info = load_base_model(view, model_ts)

        df_train = preprocess_dataframe(df_train_all, view, label2idx=label2idx)
        df_val = preprocess_dataframe(df_val_all, view, label2idx=label2idx)
        df_target = preprocess_dataframe(df_target_all, view, label2idx=None)

        ds_train = make_dataset(df_train, view, has_label=True)
        ds_val = make_dataset(df_val, view, has_label=True)
        ds_target = make_dataset(df_target, view, has_label=False)

        reuse_train = try_load_reuse_features(args.reuse_feature_cache_dir, "source", view)
        if reuse_train is not None and reuse_train[1] is not None:
            train_features, train_labels = reuse_train
        else:
            train_features, train_labels = extract_features(model, ds_train, args.infer_batch_size, has_label=True)
        maybe_save_features(args.save_feature_cache_dir, "source", view, train_features.numpy(), train_labels.numpy())

        val_features, val_labels = extract_features(model, ds_val, args.infer_batch_size, has_label=True)
        maybe_save_features(args.save_feature_cache_dir, "val", view, val_features.numpy(), val_labels.numpy())

        reuse_target = try_load_reuse_features(args.reuse_feature_cache_dir, "target", view)
        if reuse_target is not None:
            target_features, _ = reuse_target
        else:
            target_features, _ = extract_features(model, ds_target, args.infer_batch_size, has_label=False)
        maybe_save_features(args.save_feature_cache_dir, "target", view, target_features.numpy(), None)

        if args.max_train_features is not None and train_features.shape[0] > args.max_train_features:
            train_features = train_features[: args.max_train_features]
            train_labels = train_labels[: args.max_train_features]
        if args.max_val_features is not None and val_features.shape[0] > args.max_val_features:
            val_features = val_features[: args.max_val_features]
            val_labels = val_labels[: args.max_val_features]
        if args.max_target_features is not None and target_features.shape[0] > args.max_target_features:
            target_features = target_features[: args.max_target_features]
            if target_labels is not None and target_labels.shape[0] >= args.max_target_features:
                target_labels = target_labels[: args.max_target_features]

        sigma2 = compute_class_variances(train_features, train_labels, num_classes, args.variance_floor)

        mu_s = train_features.mean(dim=0)
        mu_t = target_features.mean(dim=0)
        delta, delta_raw_norm, delta_applied_norm = compute_delta(mu_s, mu_t, train_features, args.delta_normalize)

        # Freeze feature extractor: do not touch non-head params.
        for _, p in model.named_parameters():
            p.requires_grad_(False)
        head = classifier_module(model, view)
        for p in head.parameters():
            p.requires_grad_(True)

        base_val_weighted_f1, best = train_shift_robust_head(
            head,
            train_features,
            train_labels,
            val_features,
            val_labels,
            sigma2,
            delta,
            steps=args.steps,
            lr=args.lr,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            shift_rho=args.shift_rho,
            adv_radius=args.adv_radius,
            w_aug=args.w_aug,
            w_shift=args.w_shift,
            w_adv=args.w_adv,
            w_anchor=args.w_anchor,
            variance_floor=args.variance_floor,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            select_best=args.select_best,
            component_eval_subset=args.component_eval_subset,
            report_topk=args.report_topk,
        )

        # Evaluate base vs robust on target (diagnostic only).
        base_target_metrics = None
        robust_target_metrics = None
        pred_stats = None
        counts_csv_path = None
        pred_csv_path = None

        if args.eval_target_after_train:
            base_head = classifier_module(build_model_for_view(view, info), view)
            base_state = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu")
            tmp_model = build_model_for_view(view, info)
            tmp_model.load_state_dict(base_state, strict=True)
            base_head = classifier_module(tmp_model, view)
            base_head = base_head.to(device)
            robust_head = head.to(device)

            base_eval = evaluate_head_preds(base_head, target_features, target_labels if target_has_label else None, report_topk=args.report_topk)
            robust_eval = evaluate_head_preds(robust_head, target_features, target_labels if target_has_label else None, report_topk=args.report_topk)
            base_target_metrics = {k: base_eval.get(k) for k in ["acc", "weighted_f1", "macro_f1", "top2", "top3"] if k in base_eval}
            robust_target_metrics = {k: robust_eval.get(k) for k in ["acc", "weighted_f1", "macro_f1", "top2", "top3"] if k in robust_eval}

            base_probs = base_eval["probs"]
            robust_probs = robust_eval["probs"]
            base_preds = base_eval["preds"]
            robust_preds = robust_eval["preds"]

            ent_base = probs_entropy(base_probs)
            ent_robust = probs_entropy(robust_probs)
            base_counts, base_fracs = pred_distribution(base_preds, num_classes)
            robust_counts, robust_fracs = pred_distribution(robust_preds, num_classes)

            pred_stats = {
                "view": view,
                "num_samples": int(target_features.shape[0]),
                "base_target_metrics": base_target_metrics,
                "robust_target_metrics": robust_target_metrics,
                "prediction_entropy_before_after": {
                    "base": {
                        "mean_entropy": float(np.mean(ent_base)),
                        "median_entropy": float(np.median(ent_base)),
                        "p10_entropy": float(np.percentile(ent_base, 10)),
                        "p90_entropy": float(np.percentile(ent_base, 90)),
                    },
                    "robust": {
                        "mean_entropy": float(np.mean(ent_robust)),
                        "median_entropy": float(np.median(ent_robust)),
                        "p10_entropy": float(np.percentile(ent_robust, 10)),
                        "p90_entropy": float(np.percentile(ent_robust, 90)),
                    },
                },
                "prediction_count_by_class_before_after": {
                    "base_count": {idx2label[i]: int(base_counts[i]) for i in range(num_classes)},
                    "robust_count": {idx2label[i]: int(robust_counts[i]) for i in range(num_classes)},
                    "base_frac": {idx2label[i]: float(base_fracs[i]) for i in range(num_classes)},
                    "robust_frac": {idx2label[i]: float(robust_fracs[i]) for i in range(num_classes)},
                },
            }

            stats_path = os.path.join(args.output_dir, f"target_pred_stats_{view}.json")
            with open(stats_path, "w") as f:
                json.dump(pred_stats, f, indent=2, sort_keys=True)

            # counts csv
            rows = []
            for i in range(num_classes):
                rows.append(
                    {
                        "class": idx2label[i],
                        "base_count": int(base_counts[i]),
                        "robust_count": int(robust_counts[i]),
                        "base_frac": float(base_fracs[i]),
                        "robust_frac": float(robust_fracs[i]),
                    }
                )
            counts_csv_path = os.path.join(args.output_dir, f"target_pred_counts_{view}.csv")
            pd.DataFrame(rows).to_csv(counts_csv_path, index=False)

            if args.save_target_pred_csv:
                df_out = pd.DataFrame(
                    {
                        "pred_base_idx": base_preds,
                        "pred_robust_idx": robust_preds,
                        "entropy_base": ent_base,
                        "entropy_robust": ent_robust,
                        "maxprob_base": np.max(base_probs, axis=1),
                        "maxprob_robust": np.max(robust_probs, axis=1),
                    }
                )
                df_out["pred_base"] = [idx2label[int(i)] for i in df_out["pred_base_idx"].values]
                df_out["pred_robust"] = [idx2label[int(i)] for i in df_out["pred_robust_idx"].values]
                if target_has_label and target_labels is not None:
                    y_true = target_labels.detach().cpu().numpy()
                    df_out["y_label"] = y_true
                    df_out["label"] = [idx2label[int(i)] for i in y_true]
                pred_csv_path = os.path.join(args.output_dir, f"target_predictions_{view}_base_vs_robust.csv.gz")
                df_out.to_csv(pred_csv_path, index=False, compression="gzip")

        # Save robust model under new model_ts (same output dir, run_id + suffix).
        output, run_id = model_ts.split("/", 1)
        new_model_ts = f"{output}/{run_id}{args.output_suffix}"
        new_root = os.path.join("model-classifier", new_model_ts, model_name)
        ensure_dir(new_root)

        robust_info = dict(info)
        robust_info["timestamp"] = datetime.now().strftime("%Y%m%d%H%M")
        robust_info["shift_robust_classifier"] = {
            "base_model_ts": model_ts,
            "view": view,
            "steps": args.steps,
            "lr": args.lr,
            "lambda1": args.lambda1,
            "lambda2": args.lambda2,
            "shift_rho": args.shift_rho,
            "adv_radius": args.adv_radius,
            "w_aug": args.w_aug,
            "w_shift": args.w_shift,
            "w_adv": args.w_adv,
            "w_anchor": args.w_anchor,
            "variance_floor": args.variance_floor,
            "batch_size": args.batch_size,
            "infer_batch_size": args.infer_batch_size,
            "eval_every": args.eval_every,
            "select_best": args.select_best,
            "best_step": best["step"],
            "base_val_weighted_f1": float(base_val_weighted_f1),
            "best_val_weighted_f1": float(best["metrics"].get("weighted_f1", 0.0)) if best["metrics"] else 0.0,
            "delta_raw_norm": float(delta_raw_norm),
            "delta_applied_norm": float(delta_applied_norm),
            "clean_ce_at_best": None if best["components"] is None else float(best["components"]["clean_ce"]),
            "aug_ce_at_best": None if best["components"] is None else float(best["components"]["aug_ce"]),
            "shift_ce_at_best": None if best["components"] is None else float(best["components"]["shift_ce"]),
            "adv_ce_at_best": None if best["components"] is None else float(best["components"]["adv_ce"]),
            "anchor_at_best": None if best["components"] is None else float(best["components"]["anchor"]),
            "base_target_acc": None if not (target_has_label and base_target_metrics) else base_target_metrics.get("acc"),
            "robust_target_acc": None if not (target_has_label and robust_target_metrics) else robust_target_metrics.get("acc"),
            "base_target_weighted_f1": None if not (target_has_label and base_target_metrics) else base_target_metrics.get("weighted_f1"),
            "robust_target_weighted_f1": None if not (target_has_label and robust_target_metrics) else robust_target_metrics.get("weighted_f1"),
            "base_top2": None if not (target_has_label and base_target_metrics) else base_target_metrics.get("top2"),
            "robust_top2": None if not (target_has_label and robust_target_metrics) else robust_target_metrics.get("top2"),
            "base_top3": None if not (target_has_label and base_target_metrics) else base_target_metrics.get("top3"),
            "robust_top3": None if not (target_has_label and robust_target_metrics) else robust_target_metrics.get("top3"),
            "diagnostic_output_dir": args.output_dir,
            "target_pred_stats_path": None if pred_stats is None else os.path.join(args.output_dir, f"target_pred_stats_{view}.json"),
            "target_pred_counts_path": counts_csv_path,
            "target_pred_csv_path": pred_csv_path,
        }
        with open(os.path.join(new_root, "info.json"), "w") as f:
            json.dump(robust_info, f, indent=2, sort_keys=True)

        new_model_dir = model_dir_for_view(new_root, view)
        ensure_dir(new_model_dir)
        model = model.to("cpu")
        torch.save(model.state_dict(), os.path.join(new_model_dir, "pytorch_model.bin"))

        summary.setdefault("shift_robust_models", {})[view] = {
            "base_model_ts": model_ts,
            "robust_model_ts": new_model_ts,
            "best_step": best["step"],
            "base_val_weighted_f1": float(base_val_weighted_f1),
            "best_val_weighted_f1": float(best["metrics"].get("weighted_f1", 0.0)) if best["metrics"] else 0.0,
            "delta_raw_norm": float(delta_raw_norm),
            "delta_applied_norm": float(delta_applied_norm),
            "clean_ce_at_best": None if best["components"] is None else float(best["components"]["clean_ce"]),
            "aug_ce_at_best": None if best["components"] is None else float(best["components"]["aug_ce"]),
            "shift_ce_at_best": None if best["components"] is None else float(best["components"]["shift_ce"]),
            "adv_ce_at_best": None if best["components"] is None else float(best["components"]["adv_ce"]),
            "anchor_at_best": None if best["components"] is None else float(best["components"]["anchor"]),
            "target_has_label": bool(target_has_label),
            "base_target_metrics": base_target_metrics,
            "robust_target_metrics": robust_target_metrics,
            "train_features": int(train_features.shape[0]),
            "val_features": int(val_features.shape[0]),
            "target_features": int(target_features.shape[0]),
            "feature_dim": int(train_features.shape[1]),
            "num_classes": num_classes,
        }

    out_path = os.path.join("model-classifier", f"shift_robust_classifier_summary_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[ShiftRobustClassifier] Summary saved:", out_path)


if __name__ == "__main__":
    main()
