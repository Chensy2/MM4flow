import argparse
import copy
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
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


def preprocess_dataframe(df, view, label2idx):
    df = df.copy()
    df["y_label"] = df["label"].map(label2idx)
    if view in ["ps", "mm"]:
        df["ps"] = df["ps"].apply(func_ps)
    if view in ["byte", "mm"]:
        df["fwd_raw"] = df["fwd_raw"].apply(drop_empty).fillna(" ").apply(func_bytes)
        df["bwd_raw"] = df["bwd_raw"].apply(drop_empty).fillna(" ").apply(func_bytes)
    return df


def dataset_columns(view):
    if view == "ps":
        return ["ps", "y_label"]
    if view == "byte":
        return ["fwd_raw", "bwd_raw", "y_label"]
    return ["fwd_raw", "bwd_raw", "ps", "y_label"]


def tensor_columns(view):
    if view == "ps":
        return ["ps", "ps_attention_mask", "y_label"]
    if view == "byte":
        return ["raw", "raw_attention_mask", "raw_token_type_ids", "y_label"]
    return ["ps", "ps_attention_mask", "raw", "raw_attention_mask", "raw_token_type_ids", "y_label"]


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
            encoded.update({"raw": raw["input_ids"], "raw_attention_mask": raw["attention_mask"], "raw_token_type_ids": raw["token_type_ids"]})
        return encoded

    return encode


def make_dataset(df, view):
    dataset = Dataset.from_pandas(df[dataset_columns(view)])
    dataset = dataset.map(make_encoder(view), batched=True)
    remove_columns = [c for c in ["fwd_raw", "bwd_raw", "uid", "__index_level_0__"] if c in dataset.column_names]
    if remove_columns:
        dataset = dataset.remove_columns(remove_columns)
    dataset.set_format(type="torch", columns=tensor_columns(view))
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


def extract_features(model, dataset, infer_batch_size):
    model = model.to(device)
    model.eval()
    dl = torch.utils.data.DataLoader(dataset, batch_size=infer_batch_size, shuffle=False)
    feats, labels = [], []
    with torch.no_grad():
        for batch in tqdm(dl, desc="extract", unit="batch"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch, return_features=True)
            feats.append(out["features"].detach().cpu())
            labels.append(batch["y_label"].detach().cpu())
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels


def try_load_reuse_features(reuse_dir, split_name, view):
    if reuse_dir is None:
        return None
    path = os.path.join(reuse_dir, f"{split_name}_{view}.npz")
    if not os.path.exists(path):
        return None
    payload = np.load(path)
    if "features" not in payload or "labels" not in payload:
        return None
    feats = torch.from_numpy(np.asarray(payload["features"], dtype=np.float32))
    labels = torch.from_numpy(np.asarray(payload["labels"], dtype=np.int64))
    return feats, labels


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


def classifier_module(model, view):
    if view == "mm":
        return model.classifier[0]
    return model.classifier


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


def train_robust_head(
    model,
    view,
    train_features,
    train_labels,
    val_features,
    val_labels,
    steps,
    lr,
    lambda1,
    lambda2,
    aug_weight,
    anchor_weight,
    sigma2_by_class,
    batch_size,
    eval_every,
    select_best,
):
    head = classifier_module(model, view).to(device)
    head.train()

    head_orig = copy.deepcopy(head.state_dict())
    for p in head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(head.parameters(), lr=lr, eps=1e-8)

    sigma2_by_class = sigma2_by_class.to(device)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    best = {"step": -1, "metrics": None, "state": None}

    std_alpha = float(np.sqrt(lambda1)) if lambda1 > 0 else 0.0
    std_beta = float(np.sqrt(lambda2)) if lambda2 > 0 else 0.0

    rng = np.random.default_rng(42)
    n = train_features.shape[0]

    for step in range(1, steps + 1):
        idx = rng.integers(low=0, high=n, size=batch_size, endpoint=False)
        z = train_features[idx].detach()
        y = train_labels[idx]

        if std_alpha > 0:
            alpha = torch.normal(mean=1.0, std=std_alpha, size=z.shape, device=device)
        else:
            alpha = torch.ones_like(z)
        if std_beta > 0:
            beta = torch.normal(mean=0.0, std=std_beta, size=z.shape, device=device)
        else:
            beta = torch.zeros_like(z)

        sigma2 = sigma2_by_class[y].clamp_min(1e-12)
        gamma = torch.normal(mean=torch.zeros_like(z), std=torch.sqrt(sigma2))
        z_e = alpha * z + beta + gamma

        logits_clean = head(z)
        logits_aug = head(z_e)

        clean_ce = nn.functional.cross_entropy(logits_clean, y)
        aug_ce = nn.functional.cross_entropy(logits_aug, y)

        anchor = 0.0
        if anchor_weight > 0:
            for name, param in head.named_parameters():
                anchor = anchor + torch.sum((param - head_orig[name].to(device)) ** 2)

        loss = clean_ce + float(aug_weight) * aug_ce + float(anchor_weight) * anchor

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if eval_every > 0 and (step % eval_every == 0 or step == steps):
            metrics = evaluate_head(head, val_features, val_labels)
            key = select_best
            score = metrics.get(key, metrics["weighted_f1"])
            if best["metrics"] is None or score > best["metrics"].get(key, -1e9):
                best = {"step": step, "metrics": metrics, "state": copy.deepcopy(head.state_dict())}

    if best["state"] is not None:
        head.load_state_dict(best["state"])
    return best


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", required=True)
    parser.add_argument("--views", default="ps,byte,mm")
    parser.add_argument("--ps_model_ts", default=None)
    parser.add_argument("--byte_model_ts", default=None)
    parser.add_argument("--mm_model_ts", default=None)
    parser.add_argument("--output_suffix", default="_robust_cls")
    parser.add_argument("--reuse_feature_cache_dir", default=None)

    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda1", type=float, default=0.01)
    parser.add_argument("--lambda2", type=float, default=0.01)
    parser.add_argument("--aug_weight", type=float, default=1.0)
    parser.add_argument("--anchor_weight", type=float, default=1e-3)
    parser.add_argument("--variance_floor", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--select_best", choices=["weighted_f1", "macro_f1", "acc"], default="weighted_f1")

    parser.add_argument("--max_train_features", type=int, default=None)
    parser.add_argument("--max_val_features", type=int, default=None)
    args = parser.parse_args()

    views = parse_views(args.views)
    model_ts_by_view = {"ps": args.ps_model_ts, "byte": args.byte_model_ts, "mm": args.mm_model_ts}
    for v in views:
        if model_ts_by_view.get(v) is None:
            raise ValueError(f"Missing --{v}_model_ts for view={v}")

    # Load base info (use ps if available, else first view) to get label2idx.
    base_view = "ps" if "ps" in views else views[0]
    _, _, _, base_info = load_base_model(base_view, model_ts_by_view[base_view])
    label2idx = base_info["label2idx"]
    num_classes = len(label2idx)

    train_csv = os.path.join(args.source_dataset, "train.csv.gz")
    val_csv = os.path.join(args.source_dataset, "val.csv.gz")
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError(f"Missing train/val csv under {args.source_dataset}")

    df_train_all = pd.read_csv(train_csv, compression="gzip", index_col=0)
    df_val_all = pd.read_csv(val_csv, compression="gzip", index_col=0)

    summary = {
        "source_dataset": args.source_dataset,
        "views": views,
        "device": device,
        "steps": args.steps,
        "lr": args.lr,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "aug_weight": args.aug_weight,
        "anchor_weight": args.anchor_weight,
        "variance_floor": args.variance_floor,
        "batch_size": args.batch_size,
        "infer_batch_size": args.infer_batch_size,
        "eval_every": args.eval_every,
        "select_best": args.select_best,
        "output_suffix": args.output_suffix,
        "reuse_feature_cache_dir": args.reuse_feature_cache_dir,
    }

    for view in views:
        model_ts = model_ts_by_view[view]
        model, model_root, model_dir, info = load_base_model(view, model_ts)

        # Prepare datasets with base label2idx (guard against mismatched ordering).
        df_train = preprocess_dataframe(df_train_all, view, label2idx=label2idx)
        df_val = preprocess_dataframe(df_val_all, view, label2idx=label2idx)
        ds_train = make_dataset(df_train, view)
        ds_val = make_dataset(df_val, view)

        reuse_train = try_load_reuse_features(args.reuse_feature_cache_dir, "source", view)
        if reuse_train is not None:
            train_features, train_labels = reuse_train
        else:
            train_features, train_labels = extract_features(model, ds_train, args.infer_batch_size)

        val_features, val_labels = extract_features(model, ds_val, args.infer_batch_size)

        if args.max_train_features is not None and train_features.shape[0] > args.max_train_features:
            train_features = train_features[: args.max_train_features]
            train_labels = train_labels[: args.max_train_features]
        if args.max_val_features is not None and val_features.shape[0] > args.max_val_features:
            val_features = val_features[: args.max_val_features]
            val_labels = val_labels[: args.max_val_features]

        sigma2 = compute_class_variances(train_features, train_labels, num_classes, args.variance_floor)

        # Freeze feature extractor: do not touch non-head params.
        for name, p in model.named_parameters():
            p.requires_grad_(False)
        head = classifier_module(model, view)
        for p in head.parameters():
            p.requires_grad_(True)

        best = train_robust_head(
            model,
            view,
            train_features,
            train_labels,
            val_features,
            val_labels,
            steps=args.steps,
            lr=args.lr,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            aug_weight=args.aug_weight,
            anchor_weight=args.anchor_weight,
            sigma2_by_class=sigma2,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            select_best=args.select_best,
        )

        # Save robust model under new model_ts (same output dir, run_id + suffix).
        output, run_id = model_ts.split("/", 1)
        new_model_ts = f"{output}/{run_id}{args.output_suffix}"
        new_root = os.path.join("model-classifier", new_model_ts, model_name)
        ensure_dir(new_root)

        robust_info = dict(info)
        robust_info["timestamp"] = datetime.now().strftime("%Y%m%d%H%M")
        robust_info["robust_classifier"] = {
            "base_model_ts": model_ts,
            "view": view,
            "steps": args.steps,
            "lr": args.lr,
            "lambda1": args.lambda1,
            "lambda2": args.lambda2,
            "aug_weight": args.aug_weight,
            "anchor_weight": args.anchor_weight,
            "variance_floor": args.variance_floor,
            "batch_size": args.batch_size,
            "infer_batch_size": args.infer_batch_size,
            "eval_every": args.eval_every,
            "select_best": args.select_best,
            "best_step": best["step"],
            "best_metrics": best["metrics"],
        }
        with open(os.path.join(new_root, "info.json"), "w") as f:
            json.dump(robust_info, f, indent=2, sort_keys=True)

        new_model_dir = model_dir_for_view(new_root, view)
        ensure_dir(new_model_dir)
        model = model.to("cpu")
        torch.save(model.state_dict(), os.path.join(new_model_dir, "pytorch_model.bin"))

        summary.setdefault("robust_models", {})[view] = {
            "base_model_ts": model_ts,
            "robust_model_ts": new_model_ts,
            "best_step": best["step"],
            "best_metrics": best["metrics"],
            "train_features": int(train_features.shape[0]),
            "val_features": int(val_features.shape[0]),
            "feature_dim": int(train_features.shape[1]),
            "num_classes": num_classes,
        }

    out_path = os.path.join("model-classifier", f"robust_classifier_summary_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[RobustClassifier] Summary saved:", out_path)


if __name__ == "__main__":
    main()
