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
                {
                    "raw": raw["input_ids"],
                    "raw_attention_mask": raw["attention_mask"],
                    "raw_token_type_ids": raw["token_type_ids"],
                }
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
    return os.path.join(model_root, f"{view}-finetune")


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
        return feats, torch.cat(labels, dim=0)
    return feats, None


def evaluate_head(head, features, labels):
    head.eval()
    with torch.no_grad():
        logits = head(features.to(device))
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    acc = float(accuracy_score(y_true, preds))
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="weighted", zero_division=0)
    pm, rm, f1m, _ = precision_recall_fscore_support(y_true, preds, average="macro", zero_division=0)
    return {
        "acc": acc,
        "weighted_f1": float(f1),
        "macro_f1": float(f1m),
        "weighted_precision": float(p),
        "weighted_recall": float(r),
        "macro_precision": float(pm),
        "macro_recall": float(rm),
    }


def predict_head(head, features, batch_size=4096):
    head = head.to(device)
    head.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            z = features[start : start + batch_size].to(device)
            probs.append(torch.softmax(head(z), dim=1).detach().cpu())
    probs = torch.cat(probs, dim=0).numpy()
    pred = probs.argmax(axis=1).astype(np.int64)
    return probs, pred


def try_load_feature_cache(reuse_dir, split_name, view, require_labels=False, require_pred_prob=False):
    if reuse_dir is None:
        return None
    candidates = [os.path.join(reuse_dir, f"{split_name}_{view}.npz")]
    if split_name == "source_train":
        candidates.append(os.path.join(reuse_dir, f"source_{view}.npz"))
    for path in candidates:
        if not os.path.exists(path):
            continue
        payload = np.load(path)
        if "features" not in payload:
            continue
        if require_labels and "labels" not in payload:
            continue
        if require_pred_prob and ("pred" not in payload or "prob" not in payload):
            continue
        feats = torch.from_numpy(np.asarray(payload["features"], dtype=np.float32))
        labels = None
        if "labels" in payload:
            labels = torch.from_numpy(np.asarray(payload["labels"], dtype=np.int64))
        pred = np.asarray(payload["pred"], dtype=np.int64) if "pred" in payload else None
        prob = np.asarray(payload["prob"], dtype=np.float32) if "prob" in payload else None
        return {"features": feats, "labels": labels, "pred": pred, "prob": prob, "path": path}
    return None


def save_feature_cache(save_dir, split_name, view, features, labels=None, pred=None, prob=None):
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{split_name}_{view}.npz")
    payload = {"features": np.asarray(features, dtype=np.float32)}
    if labels is not None:
        payload["labels"] = np.asarray(labels, dtype=np.int64)
    if pred is not None:
        payload["pred"] = np.asarray(pred, dtype=np.int64)
    if prob is not None:
        payload["prob"] = np.asarray(prob, dtype=np.float32)
    np.savez(path, **payload)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def dataset_name_from_path(path):
    norm = os.path.normpath(path)
    base = os.path.basename(norm)
    if base == "dataset.csv.gz":
        return os.path.basename(os.path.dirname(norm))
    return base


def default_output_dir(source_dataset, target_csv, views):
    src = dataset_name_from_path(source_dataset)
    tgt = dataset_name_from_path(target_csv)
    return os.path.join("outputs", "mm4flow", f"consensus_{src}_to_{tgt}_{','.join(views)}")


def pseudo_distribution(labels, num_classes, idx2label):
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes)
    total = int(counts.sum())
    frac = counts / max(total, 1)
    entropy = float(-np.sum(frac[frac > 0] * np.log(frac[frac > 0])))
    max_ratio = float(frac.max()) if total else 0.0
    return {
        "counts": {idx2label[i]: int(counts[i]) for i in range(num_classes)},
        "fractions": {idx2label[i]: float(frac[i]) for i in range(num_classes)},
        "entropy": entropy,
        "max_class_ratio": max_ratio,
    }


def write_consensus_diagnostics(output_dir, idx2label, consensus_mask, pseudo_labels, target_known_mask, target_labels):
    rows = []
    n = len(consensus_mask)
    df_diag = pd.DataFrame(
        {
            "index": np.arange(n, dtype=np.int64),
            "is_consensus": consensus_mask.astype(bool),
            "pseudo_label_idx": np.where(consensus_mask, pseudo_labels, -1).astype(np.int64),
            "pseudo_label": [idx2label[int(i)] if keep else "" for keep, i in zip(consensus_mask, pseudo_labels)],
        }
    )
    has_diag_labels = target_known_mask is not None and target_labels is not None and bool(target_known_mask.any())
    if has_diag_labels:
        df_diag["target_known_label"] = target_known_mask.astype(bool)
        df_diag["target_label_idx"] = np.where(target_known_mask, target_labels, -1).astype(np.int64)
        df_diag["target_label"] = [idx2label[int(i)] if keep else "" for keep, i in zip(target_known_mask, target_labels)]
        df_diag["pseudo_correct_diagnostic_only"] = (
            df_diag["is_consensus"].values
            & df_diag["target_known_label"].values
            & (df_diag["pseudo_label_idx"].values == df_diag["target_label_idx"].values)
        )
    diag_path = os.path.join(output_dir, "consensus_diagnostic.csv")
    df_diag.to_csv(diag_path, index=False)

    num_classes = len(idx2label)
    for c in range(num_classes):
        cls_mask = consensus_mask & (pseudo_labels == c)
        row = {
            "class_idx": int(c),
            "class": idx2label[c],
            "consensus_per_class_count": int(cls_mask.sum()),
            "consensus_per_class_fraction": float(cls_mask.sum() / max(int(consensus_mask.sum()), 1)),
        }
        if has_diag_labels:
            known_cls = cls_mask & target_known_mask
            denom = int(known_cls.sum())
            row["consensus_per_class_precision_diagnostic_only"] = (
                None if denom == 0 else float(np.mean(target_labels[known_cls] == c))
            )
            row["known_label_count_diagnostic_only"] = denom
        rows.append(row)
    per_class_path = os.path.join(output_dir, "consensus_per_class.csv")
    pd.DataFrame(rows).to_csv(per_class_path, index=False)
    return diag_path, per_class_path


def train_consensus_head(
    model,
    view,
    source_features,
    source_labels,
    pseudo_features,
    pseudo_labels,
    val_features,
    val_labels,
    steps,
    lr,
    pseudo_weight,
    anchor_weight,
    batch_size,
    eval_every,
    select_best,
):
    head = classifier_module(model, view).to(device)
    head.train()
    head_orig = copy.deepcopy(head.state_dict())

    for p in model.parameters():
        p.requires_grad_(False)
    for p in head.parameters():
        p.requires_grad_(True)

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    non_head_trainable = [name for name, _ in trainable if "classifier" not in name]
    if non_head_trainable:
        raise RuntimeError(f"Non-classifier trainable parameters found for {view}: {non_head_trainable[:10]}")

    opt = torch.optim.Adam(head.parameters(), lr=lr, eps=1e-8)
    source_features = source_features.to(device)
    source_labels = source_labels.to(device)
    pseudo_features = pseudo_features.to(device)
    pseudo_labels = pseudo_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    rng = np.random.default_rng(42)
    n_source = int(source_features.shape[0])
    n_pseudo = int(pseudo_features.shape[0])
    best = {"step": 0, "metrics": None, "state": copy.deepcopy(head.state_dict()), "loss": None}

    if n_pseudo <= 0:
        print(f"[ConsensusPseudo] view={view}: no consensus pseudo samples; training uses source CE + anchor only.")

    for step in range(1, int(steps) + 1):
        src_idx = rng.integers(low=0, high=n_source, size=int(batch_size), endpoint=False)
        z_s = source_features[src_idx].detach()
        y_s = source_labels[src_idx]

        logits_s = head(z_s)
        source_ce = nn.functional.cross_entropy(logits_s, y_s)

        if n_pseudo > 0 and pseudo_weight > 0:
            pseudo_idx = rng.integers(low=0, high=n_pseudo, size=int(batch_size), endpoint=False)
            z_p = pseudo_features[pseudo_idx].detach()
            y_p = pseudo_labels[pseudo_idx]
            pseudo_ce = nn.functional.cross_entropy(head(z_p), y_p)
        else:
            pseudo_ce = torch.tensor(0.0, device=device)

        anchor = torch.tensor(0.0, device=device)
        if anchor_weight > 0:
            for name, param in head.named_parameters():
                anchor = anchor + torch.sum((param - head_orig[name].to(device)) ** 2)

        loss = source_ce + float(pseudo_weight) * pseudo_ce + float(anchor_weight) * anchor

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or (eval_every > 0 and step % int(eval_every) == 0) or step == int(steps):
            metrics = evaluate_head(head, val_features, val_labels)
            key = select_best
            score = metrics.get(key, metrics["weighted_f1"])
            best_score = -1e18 if best["metrics"] is None else best["metrics"].get(key, -1e18)
            if score > best_score:
                best = {
                    "step": int(step),
                    "metrics": metrics,
                    "state": copy.deepcopy(head.state_dict()),
                    "loss": {
                        "source_ce": float(source_ce.detach().cpu().item()),
                        "pseudo_ce": float(pseudo_ce.detach().cpu().item()),
                        "anchor": float(anchor.detach().cpu().item()),
                        "total": float(loss.detach().cpu().item()),
                    },
                }

    if best["state"] is not None:
        head.load_state_dict(best["state"])
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", required=True)
    parser.add_argument("--target_csv", required=True)
    parser.add_argument("--ps_model_ts", required=True)
    parser.add_argument("--byte_model_ts", required=True)
    parser.add_argument("--mm_model_ts", required=True)
    parser.add_argument("--views", default="ps,byte,mm")
    parser.add_argument("--output_suffix", default="_consensus_robust_cls")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--pseudo_weight", type=float, default=0.5)
    parser.add_argument("--anchor_weight", type=float, default=1e-3)
    parser.add_argument("--select_best", choices=["weighted_f1", "macro_f1", "acc"], default="weighted_f1")
    parser.add_argument("--min_consensus_views", type=int, default=3)
    parser.add_argument("--save_feature_cache_dir", default=None)
    parser.add_argument("--reuse_feature_cache_dir", default=None)
    parser.add_argument("--max_train_features", type=int, default=None)
    parser.add_argument("--max_val_features", type=int, default=None)
    parser.add_argument("--max_target_features", type=int, default=None)
    args = parser.parse_args()

    views = parse_views(args.views)
    if args.min_consensus_views < len(views):
        raise ValueError(
            "This first version uses hard all-selected-view agreement only. "
            f"Set --min_consensus_views to at least len(--views)={len(views)}."
        )
    if args.min_consensus_views > len(views):
        print(
            "[ConsensusPseudo] --min_consensus_views is larger than the selected view count; "
            f"using all selected views ({len(views)}) for consensus."
        )
    model_ts_by_view = {"ps": args.ps_model_ts, "byte": args.byte_model_ts, "mm": args.mm_model_ts}

    args.output_dir = args.output_dir or default_output_dir(args.source_dataset, args.target_csv, views)
    ensure_dir(args.output_dir)

    base_view = "ps" if "ps" in views else views[0]
    _, _, _, base_info = load_base_model(base_view, model_ts_by_view[base_view])
    label2idx = base_info["label2idx"]
    idx2label = {int(v): k for k, v in label2idx.items()}
    idx2label = dict(sorted(idx2label.items(), key=lambda item: item[0]))
    num_classes = len(label2idx)

    for view in views:
        _, _, _, info = load_base_model(view, model_ts_by_view[view])
        if info["label2idx"] != label2idx:
            raise ValueError(f"{view} model label2idx mapping does not match {base_view}.")

    train_csv = os.path.join(args.source_dataset, "train.csv.gz")
    val_csv = os.path.join(args.source_dataset, "val.csv.gz")
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError(f"Missing train/val csv under {args.source_dataset}")
    if not os.path.exists(args.target_csv):
        raise FileNotFoundError(f"Missing target_csv: {args.target_csv}")

    df_train_all = pd.read_csv(train_csv, compression="gzip", index_col=0)
    df_val_all = pd.read_csv(val_csv, compression="gzip", index_col=0)
    df_target_all = pd.read_csv(args.target_csv, compression="gzip", index_col=0)
    if args.max_target_features is not None and len(df_target_all) > args.max_target_features:
        df_target_all = df_target_all.head(args.max_target_features).copy()

    target_known_mask = None
    target_labels = None
    if "label" in df_target_all.columns:
        mapped = df_target_all["label"].map(label2idx)
        target_known_mask = mapped.notna().to_numpy()
        target_labels = mapped.fillna(-1).astype(np.int64).to_numpy()

    source_features_by_view = {}
    source_labels_by_view = {}
    val_features_by_view = {}
    val_labels_by_view = {}
    target_features_by_view = {}
    target_probs_by_view = {}
    target_preds_by_view = {}
    loaded_models = {}
    model_roots = {}
    model_dirs = {}
    model_infos = {}

    for view in views:
        print(f"[ConsensusPseudo] Loading/extracting view={view}")
        model, model_root, model_dir, info = load_base_model(view, model_ts_by_view[view])
        loaded_models[view] = model
        model_roots[view] = model_root
        model_dirs[view] = model_dir
        model_infos[view] = info

        df_train = preprocess_dataframe(df_train_all, view, label2idx=label2idx)
        df_val = preprocess_dataframe(df_val_all, view, label2idx=label2idx)
        df_target = preprocess_dataframe(df_target_all, view, label2idx=None)

        reuse_train = try_load_feature_cache(args.reuse_feature_cache_dir, "source_train", view, require_labels=True)
        if reuse_train is None:
            ds_train = make_dataset(df_train, view, has_label=True)
            train_features, train_labels = extract_features(model, ds_train, args.infer_batch_size, has_label=True)
            save_feature_cache(
                args.save_feature_cache_dir, "source_train", view, train_features.numpy(), labels=train_labels.numpy()
            )
        else:
            train_features, train_labels = reuse_train["features"], reuse_train["labels"]

        reuse_val = try_load_feature_cache(args.reuse_feature_cache_dir, "source_val", view, require_labels=True)
        if reuse_val is None:
            ds_val = make_dataset(df_val, view, has_label=True)
            val_features, val_labels = extract_features(model, ds_val, args.infer_batch_size, has_label=True)
            save_feature_cache(args.save_feature_cache_dir, "source_val", view, val_features.numpy(), labels=val_labels.numpy())
        else:
            val_features, val_labels = reuse_val["features"], reuse_val["labels"]

        if args.max_train_features is not None and train_features.shape[0] > args.max_train_features:
            train_features = train_features[: args.max_train_features]
            train_labels = train_labels[: args.max_train_features]
        if args.max_val_features is not None and val_features.shape[0] > args.max_val_features:
            val_features = val_features[: args.max_val_features]
            val_labels = val_labels[: args.max_val_features]

        reuse_target = try_load_feature_cache(args.reuse_feature_cache_dir, "target", view, require_pred_prob=True)
        if reuse_target is None:
            ds_target = make_dataset(df_target, view, has_label=False)
            target_features, _ = extract_features(model, ds_target, args.infer_batch_size, has_label=False)
            head = classifier_module(model, view)
            target_prob, target_pred = predict_head(head, target_features)
            save_feature_cache(
                args.save_feature_cache_dir,
                "target",
                view,
                target_features.numpy(),
                pred=target_pred,
                prob=target_prob,
            )
        else:
            target_features = reuse_target["features"]
            target_pred = reuse_target["pred"]
            target_prob = reuse_target["prob"]

        source_features_by_view[view] = train_features
        source_labels_by_view[view] = train_labels
        val_features_by_view[view] = val_features
        val_labels_by_view[view] = val_labels
        target_features_by_view[view] = target_features
        target_probs_by_view[view] = target_prob
        target_preds_by_view[view] = target_pred

    target_len = min(len(target_preds_by_view[v]) for v in views)
    for view in views:
        if len(target_preds_by_view[view]) != target_len:
            print(f"[ConsensusPseudo] Warning: truncating {view} target outputs to common length {target_len}.")
            target_features_by_view[view] = target_features_by_view[view][:target_len]
            target_probs_by_view[view] = target_probs_by_view[view][:target_len]
            target_preds_by_view[view] = target_preds_by_view[view][:target_len]
    if target_known_mask is not None:
        target_known_mask = target_known_mask[:target_len]
        target_labels = target_labels[:target_len]

    pred_stack = np.stack([target_preds_by_view[v] for v in views], axis=1)
    consensus_mask = np.all(pred_stack == pred_stack[:, :1], axis=1)
    pseudo_labels_all = pred_stack[:, 0].astype(np.int64)
    pseudo_labels_consensus = pseudo_labels_all[consensus_mask]
    num_consensus = int(consensus_mask.sum())
    dist = pseudo_distribution(pseudo_labels_consensus, num_classes, idx2label) if num_consensus else {
        "counts": {idx2label[i]: 0 for i in range(num_classes)},
        "fractions": {idx2label[i]: 0.0 for i in range(num_classes)},
        "entropy": 0.0,
        "max_class_ratio": 0.0,
    }

    consensus_pseudo_acc = None
    if target_known_mask is not None and bool((consensus_mask & target_known_mask).any()):
        known_consensus = consensus_mask & target_known_mask
        consensus_pseudo_acc = float(np.mean(pseudo_labels_all[known_consensus] == target_labels[known_consensus]))
    diag_path, per_class_path = write_consensus_diagnostics(
        args.output_dir, idx2label, consensus_mask, pseudo_labels_all, target_known_mask, target_labels
    )

    summary = {
        "method": "consensus_pseudo_classifier_finetune",
        "source_dataset": args.source_dataset,
        "target_csv": args.target_csv,
        "views_for_consensus": views,
        "num_target_samples": int(target_len),
        "num_consensus_samples": num_consensus,
        "consensus_ratio": float(num_consensus / max(target_len, 1)),
        "pseudo_label_distribution": dist["counts"],
        "pseudo_label_fraction": dist["fractions"],
        "pseudo_label_entropy": dist["entropy"],
        "max_pseudo_class_ratio": dist["max_class_ratio"],
        "target_label_diagnostics_used_for_training": False,
        "consensus_pseudo_acc_diagnostic_only": consensus_pseudo_acc,
        "consensus_diagnostic_csv": diag_path,
        "consensus_per_class_csv": per_class_path,
        "pseudo_weight": args.pseudo_weight,
        "anchor_weight": args.anchor_weight,
        "steps": args.steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "infer_batch_size": args.infer_batch_size,
        "eval_every": args.eval_every,
        "select_best": args.select_best,
        "output_suffix": args.output_suffix,
        "output_dir": args.output_dir,
        "reuse_feature_cache_dir": args.reuse_feature_cache_dir,
        "save_feature_cache_dir": args.save_feature_cache_dir,
        "device": device,
        "per_view": {},
    }

    for view in views:
        print(f"[ConsensusPseudo] Fine-tuning classifier head for view={view}")
        model = loaded_models[view]
        model_ts = model_ts_by_view[view]
        for p in model.parameters():
            p.requires_grad_(False)
        head = classifier_module(model, view)
        for p in head.parameters():
            p.requires_grad_(True)

        pseudo_features = target_features_by_view[view][consensus_mask]
        pseudo_labels = torch.from_numpy(pseudo_labels_all[consensus_mask].astype(np.int64))
        best = train_consensus_head(
            model=model,
            view=view,
            source_features=source_features_by_view[view],
            source_labels=source_labels_by_view[view],
            pseudo_features=pseudo_features,
            pseudo_labels=pseudo_labels,
            val_features=val_features_by_view[view],
            val_labels=val_labels_by_view[view],
            steps=args.steps,
            lr=args.lr,
            pseudo_weight=args.pseudo_weight,
            anchor_weight=args.anchor_weight,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            select_best=args.select_best,
        )

        output, run_id = model_ts.split("/", 1)
        new_model_ts = f"{output}/{run_id}{args.output_suffix}"
        new_root = os.path.join("model-classifier", new_model_ts, model_name)
        ensure_dir(new_root)
        new_model_dir = model_dir_for_view(new_root, view)
        ensure_dir(new_model_dir)

        info = dict(model_infos[view])
        info["timestamp"] = datetime.now().strftime("%Y%m%d%H%M")
        info["consensus_pseudo_classifier"] = {
            "enabled": True,
            "base_model_ts": model_ts,
            "source_dataset": args.source_dataset,
            "target_csv": args.target_csv,
            "view": view,
            "views_for_consensus": views,
            "num_target_samples": int(target_len),
            "num_consensus_samples": num_consensus,
            "consensus_ratio": float(num_consensus / max(target_len, 1)),
            "pseudo_weight": args.pseudo_weight,
            "anchor_weight": args.anchor_weight,
            "steps": args.steps,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "infer_batch_size": args.infer_batch_size,
            "eval_every": args.eval_every,
            "select_best": args.select_best,
            "best_step": best["step"],
            "best_metrics": best["metrics"],
            "best_loss": best["loss"],
            "target_label_diagnostics_used_for_training": False,
            "consensus_pseudo_acc_diagnostic_only": consensus_pseudo_acc,
        }
        with open(os.path.join(new_root, "info.json"), "w") as f:
            json.dump(info, f, indent=2, sort_keys=True)

        model = model.to("cpu")
        torch.save(model.state_dict(), os.path.join(new_model_dir, "pytorch_model.bin"))

        summary["per_view"][view] = {
            "base_model_ts": model_ts,
            "output_model_ts": new_model_ts,
            "best_step": int(best["step"]),
            "best_val_weighted_f1": None if best["metrics"] is None else best["metrics"].get("weighted_f1"),
            "best_metrics": best["metrics"],
            "best_loss": best["loss"],
            "feature_dim": int(source_features_by_view[view].shape[1]),
            "source_features": int(source_features_by_view[view].shape[0]),
            "val_features": int(val_features_by_view[view].shape[0]),
            "pseudo_features": int(pseudo_features.shape[0]),
        }

    summary_path = os.path.join(args.output_dir, "consensus_pseudo_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    global_summary_path = os.path.join(
        "model-classifier", f"consensus_pseudo_classifier_summary_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    )
    with open(global_summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[ConsensusPseudo] Summary saved:", summary_path)
    print("[ConsensusPseudo] Global summary saved:", global_summary_path)


if __name__ == "__main__":
    main()
