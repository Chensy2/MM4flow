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


def _load_npz_features(path, require_labels):
    payload = np.load(path)
    if "features" not in payload:
        return None
    feats = torch.from_numpy(np.asarray(payload["features"], dtype=np.float32))
    labels = None
    if require_labels:
        if "labels" not in payload:
            return None
        labels = torch.from_numpy(np.asarray(payload["labels"], dtype=np.int64))
    return feats, labels


def try_load_reuse_features(reuse_dir, split_name, view, require_labels):
    if reuse_dir is None:
        return None
    # Prefer new naming.
    candidates = [
        os.path.join(reuse_dir, f"{split_name}_{view}.npz"),
    ]
    # Backward-compatible names (shift-robust style).
    legacy_map = {
        "source_train": f"source_{view}.npz",
        "source_val": f"val_{view}.npz",
        "target": f"target_{view}.npz",
    }
    if split_name in legacy_map:
        candidates.append(os.path.join(reuse_dir, legacy_map[split_name]))
    for path in candidates:
        if os.path.exists(path):
            out = _load_npz_features(path, require_labels=require_labels)
            if out is not None:
                return out
    return None


def maybe_save_features(save_dir, split_name, view, features, labels):
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{split_name}_{view}.npz")
    payload = {"features": np.asarray(features, dtype=np.float32)}
    if labels is not None:
        payload["labels"] = np.asarray(labels, dtype=np.int64)
    np.savez(path, **payload)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _l2_normalize(x, eps=1e-12):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def assert_finite_array(name, arr):
    arr_np = np.asarray(arr)
    finite = np.isfinite(arr_np)
    if not np.all(finite):
        bad = int(arr_np.size - np.sum(finite))
        total = int(arr_np.size)
        raise ValueError(f"[CCT] Non-finite values in {name}: {bad}/{total}. Check feature extraction/cache.")


def pairwise_distance_min_to_protos(query, protos, metric):
    # query: [K,D], protos: [P,D] for one class
    if protos.shape[0] == 0:
        return np.full((query.shape[0],), np.inf, dtype=np.float32)
    if metric == "euclidean":
        # (a-b)^2 = a^2 + b^2 - 2ab
        q2 = np.sum(query * query, axis=1, keepdims=True)
        p2 = np.sum(protos * protos, axis=1, keepdims=True).T
        d2 = q2 + p2 - 2.0 * (query @ protos.T)
        d2 = np.maximum(d2, 0.0)
        return np.sqrt(np.min(d2, axis=1)).astype(np.float32)
    # cosine
    qn = _l2_normalize(query.astype(np.float64))
    pn = _l2_normalize(protos.astype(np.float64))
    sim = qn @ pn.T
    dist = 1.0 - sim
    return np.min(dist, axis=1).astype(np.float32)


def point_to_center_distance(points, center, metric):
    assert_finite_array("point_to_center_distance.points", points)
    assert_finite_array("point_to_center_distance.center", center)
    if metric == "euclidean":
        return np.linalg.norm(points - center[None, :], axis=1)
    pn = _l2_normalize(points.astype(np.float64))
    cn = center.astype(np.float64) / max(1e-12, np.linalg.norm(center))
    return 1.0 - (pn @ cn.reshape(-1, 1)).reshape(-1)


def dpmeans_class(features, metric, dp_lambda, dp_percentile, min_proto_mass, max_iter=30):
    # features: [N,D]
    if features.shape[0] == 0:
        return np.zeros((0, features.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), 0.0
    center = np.mean(features, axis=0)
    if dp_lambda == "auto":
        dist = point_to_center_distance(features, center, metric)
        lam = float(np.percentile(dist, dp_percentile))
        lam = max(lam, 1e-6)
    else:
        lam = float(dp_lambda)
        lam = max(lam, 1e-12)

    protos = [center.astype(np.float32)]
    assign = None

    for _ in range(max_iter):
        # assign each point
        proto_arr = np.stack(protos, axis=0)
        if metric == "euclidean":
            dists = np.linalg.norm(features[:, None, :] - proto_arr[None, :, :], axis=2)
        else:
            fn = _l2_normalize(features.astype(np.float64))
            pn = _l2_normalize(proto_arr.astype(np.float64))
            dists = 1.0 - (fn @ pn.T)
        nearest = np.argmin(dists, axis=1)
        nearest_dist = dists[np.arange(features.shape[0]), nearest]
        # create new prototypes for far points (one-by-one in order)
        created = False
        for i in range(features.shape[0]):
            if float(nearest_dist[i]) > lam:
                protos.append(features[i].astype(np.float32))
                created = True
        proto_arr = np.stack(protos, axis=0)
        # reassign after possible creation
        if metric == "euclidean":
            dists = np.linalg.norm(features[:, None, :] - proto_arr[None, :, :], axis=2)
        else:
            fn = _l2_normalize(features.astype(np.float64))
            pn = _l2_normalize(proto_arr.astype(np.float64))
            dists = 1.0 - (fn @ pn.T)
        new_assign = np.argmin(dists, axis=1).astype(np.int64)
        if assign is not None and (not created) and np.all(new_assign == assign):
            assign = new_assign
            break
        assign = new_assign
        # recompute prototypes
        new_protos = []
        for k in range(proto_arr.shape[0]):
            idx = np.where(assign == k)[0]
            if len(idx) == 0:
                continue
            new_protos.append(np.mean(features[idx], axis=0).astype(np.float32))
        if not new_protos:
            protos = [center.astype(np.float32)]
        else:
            protos = new_protos

    proto_arr = np.stack(protos, axis=0)
    # final assign/mass
    if metric == "euclidean":
        dists = np.linalg.norm(features[:, None, :] - proto_arr[None, :, :], axis=2)
    else:
        fn = _l2_normalize(features.astype(np.float64))
        pn = _l2_normalize(proto_arr.astype(np.float64))
        dists = 1.0 - (fn @ pn.T)
    assign = np.argmin(dists, axis=1).astype(np.int64)
    masses = np.bincount(assign, minlength=proto_arr.shape[0]).astype(np.int64)

    # merge small prototypes
    if proto_arr.shape[0] > 1:
        large = np.where(masses >= int(min_proto_mass))[0]
        small = np.where(masses < int(min_proto_mass))[0]
        if len(large) == 0:
            proto_arr = center.astype(np.float32)[None, :]
            assign = np.zeros((features.shape[0],), dtype=np.int64)
            masses = np.array([features.shape[0]], dtype=np.int64)
        elif len(small) > 0:
            # reassign points in small to nearest large prototype
            large_protos = proto_arr[large]
            if metric == "euclidean":
                d_large = np.linalg.norm(features[:, None, :] - large_protos[None, :, :], axis=2)
            else:
                fn = _l2_normalize(features.astype(np.float64))
                ln = _l2_normalize(large_protos.astype(np.float64))
                d_large = 1.0 - (fn @ ln.T)
            assign_large = large[np.argmin(d_large, axis=1)]
            assign = np.array([int(np.where(large == a)[0][0]) for a in assign_large], dtype=np.int64)
            # recompute prototypes/masses on merged clusters
            new_protos = []
            for k in range(len(large)):
                idx = np.where(assign == k)[0]
                new_protos.append(np.mean(features[idx], axis=0).astype(np.float32))
            proto_arr = np.stack(new_protos, axis=0)
            masses = np.bincount(assign, minlength=proto_arr.shape[0]).astype(np.int64)

    return proto_arr.astype(np.float32), masses.astype(np.int64), assign.astype(np.int64), lam


def kmeans_target(features, K, metric, max_iter=50, seed=42):
    assert_finite_array("target_features_for_kmeans", features)
    # Prefer sklearn if available.
    try:
        from sklearn.cluster import KMeans  # type: ignore

        km = KMeans(n_clusters=K, n_init=10, random_state=seed)
        ids = km.fit_predict(features)
        centers = km.cluster_centers_
        masses = np.bincount(ids, minlength=K).astype(np.int64)
        return centers.astype(np.float32), ids.astype(np.int64), masses
    except Exception:
        pass

    rng = np.random.default_rng(seed)
    n, d = features.shape
    if K <= 1:
        centers = np.mean(features, axis=0, keepdims=True)
        ids = np.zeros((n,), dtype=np.int64)
        masses = np.array([n], dtype=np.int64)
        return centers.astype(np.float32), ids, masses

    # kmeans++ init
    centers = np.zeros((K, d), dtype=np.float32)
    first = rng.integers(0, n)
    centers[0] = features[first]
    if metric == "euclidean":
        dist = np.linalg.norm(features - centers[0][None, :], axis=1) ** 2
    else:
        fn = _l2_normalize(features.astype(np.float64))
        c0 = centers[0].astype(np.float64) / max(1e-12, np.linalg.norm(centers[0]))
        dist = (1.0 - (fn @ c0.reshape(-1, 1)).reshape(-1)) ** 2
    for k in range(1, K):
        probs = dist / max(1e-12, np.sum(dist))
        idx = int(rng.choice(n, p=probs))
        centers[k] = features[idx]
        if metric == "euclidean":
            dnew = np.linalg.norm(features - centers[k][None, :], axis=1) ** 2
        else:
            ck = centers[k].astype(np.float64) / max(1e-12, np.linalg.norm(centers[k]))
            dnew = (1.0 - (fn @ ck.reshape(-1, 1)).reshape(-1)) ** 2
        dist = np.minimum(dist, dnew)

    ids = np.zeros((n,), dtype=np.int64)
    for _ in range(max_iter):
        prev = ids.copy()
        if metric == "euclidean":
            dists = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        else:
            fn = _l2_normalize(features.astype(np.float64))
            cn = _l2_normalize(centers.astype(np.float64))
            dists = 1.0 - (fn @ cn.T)
        ids = np.argmin(dists, axis=1).astype(np.int64)
        if np.all(ids == prev):
            break
        for k in range(K):
            idx = np.where(ids == k)[0]
            if len(idx) == 0:
                centers[k] = features[int(rng.integers(0, n))]
            else:
                centers[k] = np.mean(features[idx], axis=0)
    masses = np.bincount(ids, minlength=K).astype(np.int64)
    return centers.astype(np.float32), ids.astype(np.int64), masses


def softmax_np(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def compute_alignment(target_centers, target_masses, source_protos_by_class, metric, tau):
    assert_finite_array("target_cluster_centers", target_centers)
    assert_finite_array("target_cluster_masses", target_masses)
    C = len(source_protos_by_class)
    K = target_centers.shape[0]
    dist = np.zeros((K, C), dtype=np.float32)
    for c in range(C):
        assert_finite_array(f"source_prototypes_class_{c}", source_protos_by_class[c])
        dist[:, c] = pairwise_distance_min_to_protos(target_centers, source_protos_by_class[c], metric)
    assert_finite_array("cluster_class_distance", dist)
    A = softmax_np(-dist / max(1e-12, float(tau)), axis=1).astype(np.float32)
    assert_finite_array("cluster_class_assignment", A)
    ent = -np.sum(A * np.log(A + 1e-12), axis=1)
    # class-wise mass and mu_target
    mass_c = (A * target_masses[:, None]).sum(axis=0)  # [C]
    weighted_sum = (A * target_masses[:, None]).T @ target_centers  # [C,D]
    mu_target = weighted_sum / np.maximum(mass_c[:, None], 1e-12)
    return A, ent.astype(np.float32), mu_target.astype(np.float32)


def evaluate_head(head, features, labels):
    head.eval()
    with torch.no_grad():
        logits = head(features.to(device))
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    acc = float(accuracy_score(y_true, preds))
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="weighted", zero_division=0)
    return {"acc": acc, "weighted_f1": float(f1), "weighted_precision": float(p), "weighted_recall": float(r)}


def evaluate_head_probs(head, features, labels=None, report_topk=3):
    head.eval()
    with torch.no_grad():
        logits = head(features.to(device))
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1).astype(np.int64)
    out = {"preds": preds, "probs": probs}
    if labels is None:
        return out
    y_true = labels.detach().cpu().numpy().astype(np.int64)
    out["acc"] = float(accuracy_score(y_true, preds))
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="weighted", zero_division=0)
    out["weighted_f1"] = float(f1)
    out["weighted_precision"] = float(p)
    out["weighted_recall"] = float(r)
    if report_topk and report_topk >= 2:
        topk = min(int(report_topk), probs.shape[1])
        top_idx = np.argsort(-probs, axis=1)[:, :topk]
        y_col = y_true.reshape(-1, 1)
        if topk >= 2:
            out["top2"] = float(np.mean(np.any(top_idx[:, :2] == y_col, axis=1)))
        if topk >= 3:
            out["top3"] = float(np.mean(np.any(top_idx[:, :3] == y_col, axis=1)))
    return out


def probs_entropy(probs):
    p = np.asarray(probs, dtype=np.float64)
    return -np.sum(p * np.log(p + 1e-12), axis=1)


def pred_distribution(preds, num_classes):
    preds = np.asarray(preds, dtype=np.int64)
    counts = np.bincount(preds, minlength=num_classes).astype(np.int64)
    fracs = counts.astype(np.float64) / max(1, preds.shape[0])
    return counts, fracs


def train_cct_head(
    head,
    train_features,
    train_labels,
    val_features,
    val_labels,
    delta_by_class,
    *,
    steps,
    lr,
    batch_size,
    eval_every,
    select_best,
    shift_weight,
    anchor_weight,
    cct_shift_rho,
):
    head = head.to(device)
    head.train()
    head_orig_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
    opt = torch.optim.AdamW(head.parameters(), lr=lr)

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    delta_by_class = delta_by_class.to(device)

    rng = np.random.default_rng(42)
    n = train_features.shape[0]

    base_val = evaluate_head(head, val_features, val_labels)
    base_val_weighted_f1 = float(base_val["weighted_f1"])

    best = {"step": -1, "metrics": None, "state": None}

    for step in range(1, steps + 1):
        idx = rng.integers(low=0, high=n, size=batch_size, endpoint=False)
        z = train_features[idx].detach()
        y = train_labels[idx]
        z_shift = z + float(cct_shift_rho) * delta_by_class[y]

        logits_clean = head(z)
        logits_shift = head(z_shift)
        clean_ce = nn.functional.cross_entropy(logits_clean, y)
        shift_ce = nn.functional.cross_entropy(logits_shift, y)

        anchor = 0.0
        if anchor_weight > 0:
            for name, param in head.named_parameters():
                anchor = anchor + torch.sum((param - head_orig_state[name].to(device)) ** 2)

        loss = clean_ce + float(shift_weight) * shift_ce + float(anchor_weight) * anchor
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if eval_every > 0 and (step % eval_every == 0 or step == steps):
            metrics = evaluate_head(head, val_features, val_labels)
            score = float(metrics.get(select_best, metrics["weighted_f1"]))
            if best["metrics"] is None or score > float(best["metrics"].get(select_best, -1e9)):
                best = {"step": step, "metrics": metrics, "state": copy.deepcopy({k: v.detach().cpu().clone() for k, v in head.state_dict().items()})}

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
    parser.add_argument("--output_suffix", default="_cct_robust_cls")
    parser.add_argument("--output_dir", default=None)

    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--select_best", choices=["weighted_f1", "acc"], default="weighted_f1")
    parser.add_argument("--anchor_weight", type=float, default=1e-3)
    parser.add_argument("--shift_weight", type=float, default=1.0)
    parser.add_argument("--cct_shift_rho", type=float, default=0.5)

    parser.add_argument("--cct_target_cluster_ratio", type=float, default=2.0)
    parser.add_argument("--cct_assignment_tau", type=float, default=0.2)
    parser.add_argument("--cct_dp_lambda", default="auto")
    parser.add_argument("--cct_dp_percentile", type=float, default=75.0)
    parser.add_argument("--cct_distance", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--cct_min_proto_mass", type=int, default=2)

    parser.add_argument("--report_topk", type=int, default=3)
    parser.add_argument("--eval_target_after_train", action="store_true")
    parser.add_argument("--save_target_pred_csv", action="store_true")

    parser.add_argument("--reuse_feature_cache_dir", default=None)
    parser.add_argument("--save_feature_cache_dir", default=None)
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
        args.output_dir = os.path.join("outputs", "mm4flow", f"cct_{stamp}")
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
    # NOTE: do not use target labels even if present.

    summary = {
        "source_dataset": args.source_dataset,
        "target_csv": args.target_csv,
        "views": views,
        "device": device,
        "steps": args.steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "infer_batch_size": args.infer_batch_size,
        "eval_every": args.eval_every,
        "select_best": args.select_best,
        "anchor_weight": args.anchor_weight,
        "shift_weight": args.shift_weight,
        "cct_shift_rho": args.cct_shift_rho,
        "cct_target_cluster_ratio": args.cct_target_cluster_ratio,
        "cct_assignment_tau": args.cct_assignment_tau,
        "cct_dp_lambda": args.cct_dp_lambda,
        "cct_dp_percentile": args.cct_dp_percentile,
        "cct_distance": args.cct_distance,
        "cct_min_proto_mass": args.cct_min_proto_mass,
        "report_topk": int(args.report_topk),
        "eval_target_after_train": bool(args.eval_target_after_train),
        "save_target_pred_csv": bool(args.save_target_pred_csv),
        "output_suffix": args.output_suffix,
        "reuse_feature_cache_dir": args.reuse_feature_cache_dir,
        "save_feature_cache_dir": args.save_feature_cache_dir,
        "output_dir": args.output_dir,
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

        reuse_train = try_load_reuse_features(args.reuse_feature_cache_dir, "source_train", view, require_labels=True)
        if reuse_train is not None:
            train_features, train_labels = reuse_train
        else:
            train_features, train_labels = extract_features(model, ds_train, args.infer_batch_size, has_label=True)
        maybe_save_features(
            args.save_feature_cache_dir, "source_train", view, train_features.numpy(), train_labels.numpy()
        )

        reuse_val = try_load_reuse_features(args.reuse_feature_cache_dir, "source_val", view, require_labels=True)
        if reuse_val is not None:
            val_features, val_labels = reuse_val
        else:
            val_features, val_labels = extract_features(model, ds_val, args.infer_batch_size, has_label=True)
        maybe_save_features(args.save_feature_cache_dir, "source_val", view, val_features.numpy(), val_labels.numpy())

        reuse_target = try_load_reuse_features(args.reuse_feature_cache_dir, "target", view, require_labels=False)
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

        # If target has labels, use ONLY for diagnostics (never for training/selection).
        target_has_label = "label" in df_target_all.columns
        target_labels = None
        if target_has_label:
            df_label = df_target_all[df_target_all["label"].isin(label2idx.keys())].copy()
            if len(df_label) > 0:
                target_labels = torch.from_numpy(df_label["label"].map(label2idx).astype(np.int64).values)
            else:
                target_has_label = False
                target_labels = None

        train_np = train_features.numpy().astype(np.float32)
        y_np = train_labels.numpy()
        if y_np.ndim != 1:
            y_np = y_np.reshape(-1)
        if np.any(pd.isna(y_np)):
            bad = int(np.sum(pd.isna(y_np)))
            raise ValueError(f"[CCT] Found {bad} NaN labels in source_train for view={view}. Check label2idx mapping.")
        y_np = y_np.astype(np.int64)
        val_np = val_features
        val_y = val_labels
        target_np = target_features.numpy().astype(np.float32)
        assert_finite_array(f"{view}.source_train_features", train_np)
        assert_finite_array(f"{view}.source_val_features", val_features.numpy())
        assert_finite_array(f"{view}.target_features", target_np)

        # Step 2: class-wise DPMeans
        source_protos_by_class = []
        source_proto_counts = []
        source_mu = np.zeros((num_classes, train_np.shape[1]), dtype=np.float32)
        proto_pack = []
        proto_class_ids = []
        proto_masses_pack = []
        dp_lams = []

        empty_classes = []
        for c in tqdm(range(num_classes), desc=f"dpmeans[{view}]", unit="class"):
            idx = np.where(y_np == c)[0]
            if idx.size == 0:
                empty_classes.append(int(c))
                # Fallback: zeros
                mu_c = np.zeros((train_np.shape[1],), dtype=np.float32)
                protos_c = mu_c[None, :]
                masses_c = np.array([0], dtype=np.int64)
                lam_c = 1e-6
            else:
                feats_c = train_np[idx]
                mu_c = np.mean(feats_c, axis=0).astype(np.float32)
                protos_c, masses_c, _, lam_c = dpmeans_class(
                    feats_c,
                    metric=args.cct_distance,
                    dp_lambda=args.cct_dp_lambda,
                    dp_percentile=args.cct_dp_percentile,
                    min_proto_mass=args.cct_min_proto_mass,
                    max_iter=30,
                )
            source_mu[c] = mu_c
            source_protos_by_class.append(protos_c.astype(np.float32))
            source_proto_counts.append(int(protos_c.shape[0]))
            dp_lams.append(float(lam_c))

            for m in range(protos_c.shape[0]):
                proto_pack.append(protos_c[m])
                proto_class_ids.append(c)
                proto_masses_pack.append(int(masses_c[m]) if m < len(masses_c) else 0)

        source_proto_counts_arr = np.asarray(source_proto_counts, dtype=np.int64)

        # Step 3: target over-clustering
        K = int(round(float(args.cct_target_cluster_ratio) * float(num_classes)))
        K = max(1, min(K, int(target_np.shape[0])))
        target_centers, target_ids, target_masses = kmeans_target(target_np, K, metric=args.cct_distance)

        # Step 4/5: alignment and delta
        A, ent, mu_target = compute_alignment(
            target_centers.astype(np.float32),
            target_masses.astype(np.float32),
            source_protos_by_class,
            metric=args.cct_distance,
            tau=args.cct_assignment_tau,
        )
        delta = (mu_target - source_mu).astype(np.float32)  # [C,D]

        delta_norms = np.linalg.norm(delta, axis=1)
        assignment_entropy_stats = {
            "assignment_entropy_min": float(np.min(ent)) if ent.size else 0.0,
            "assignment_entropy_mean": float(np.mean(ent)) if ent.size else 0.0,
            "assignment_entropy_max": float(np.max(ent)) if ent.size else 0.0,
        }
        delta_norm_stats = {
            "delta_norm_min": float(np.min(delta_norms)) if delta_norms.size else 0.0,
            "delta_norm_mean": float(np.mean(delta_norms)) if delta_norms.size else 0.0,
            "delta_norm_max": float(np.max(delta_norms)) if delta_norms.size else 0.0,
        }

        # Step 6: head-only finetune
        for _, p in model.named_parameters():
            p.requires_grad_(False)
        head = classifier_module(model, view)
        for p in head.parameters():
            p.requires_grad_(True)

        base_head_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
        delta_torch = torch.from_numpy(delta)
        base_val_weighted_f1, best = train_cct_head(
            head,
            train_features,
            train_labels,
            val_features,
            val_labels,
            delta_torch,
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            select_best=args.select_best,
            shift_weight=args.shift_weight,
            anchor_weight=args.anchor_weight,
            cct_shift_rho=args.cct_shift_rho,
        )

        # Target diagnostics: base vs robust (for analysis only; no target labels used in training/selection).
        base_target_metrics = None
        robust_target_metrics = None
        counts_csv_path = None
        pred_csv_path = None
        stats_path = None
        if args.eval_target_after_train:
            base_head = copy.deepcopy(head).to(device)
            base_head.load_state_dict(base_head_state, strict=True)
            robust_head = head.to(device)

            base_eval = evaluate_head_probs(base_head, target_features, target_labels if target_has_label else None, report_topk=args.report_topk)
            robust_eval = evaluate_head_probs(robust_head, target_features, target_labels if target_has_label else None, report_topk=args.report_topk)
            base_probs = base_eval["probs"]
            robust_probs = robust_eval["probs"]
            base_preds = base_eval["preds"]
            robust_preds = robust_eval["preds"]

            ent_base = probs_entropy(base_probs)
            ent_robust = probs_entropy(robust_probs)
            base_counts, base_fracs = pred_distribution(base_preds, num_classes)
            robust_counts, robust_fracs = pred_distribution(robust_preds, num_classes)

            if target_has_label:
                base_target_metrics = {k: base_eval.get(k) for k in ["acc", "weighted_f1", "top2", "top3"] if k in base_eval}
                robust_target_metrics = {k: robust_eval.get(k) for k in ["acc", "weighted_f1", "top2", "top3"] if k in robust_eval}

            pred_stats = {
                "view": view,
                "num_samples": int(target_features.shape[0]),
                "target_has_label": bool(target_has_label),
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

        # diagnostics save
        view_prefix = os.path.join(args.output_dir, f"cct_{view}")
        np.save(view_prefix + "_delta.npy", delta.astype(np.float32))
        np.save(view_prefix + "_assignment.npy", A.astype(np.float32))
        np.save(view_prefix + "_target_cluster_centers.npy", target_centers.astype(np.float32))
        np.save(view_prefix + "_target_cluster_masses.npy", target_masses.astype(np.int64))
        np.save(view_prefix + "_source_prototypes.npy", np.asarray(proto_pack, dtype=np.float32))
        np.save(view_prefix + "_source_prototype_class_ids.npy", np.asarray(proto_class_ids, dtype=np.int64))
        np.save(view_prefix + "_source_prototype_masses.npy", np.asarray(proto_masses_pack, dtype=np.int64))

        cct_summary_view = {
            "view": view,
            "base_model_ts": model_ts,
            "source_dataset": args.source_dataset,
            "target_csv": args.target_csv,
            "feature_dim": int(train_np.shape[1]),
            "num_classes": num_classes,
            "target_cluster_count": int(K),
            "empty_source_classes": empty_classes,
            "dp_lambda_by_class": dp_lams,
            "source_proto_count_by_class": source_proto_counts,
            "source_proto_count_min": int(source_proto_counts_arr.min()) if source_proto_counts_arr.size else 0,
            "source_proto_count_mean": float(source_proto_counts_arr.mean()) if source_proto_counts_arr.size else 0.0,
            "source_proto_count_max": int(source_proto_counts_arr.max()) if source_proto_counts_arr.size else 0,
            **delta_norm_stats,
            **assignment_entropy_stats,
            "base_val_weighted_f1": float(base_val_weighted_f1),
            "best_step": int(best["step"]),
            "best_val_weighted_f1": float(best["metrics"]["weighted_f1"]) if best["metrics"] else 0.0,
            "base_target_acc": None if not (target_has_label and base_target_metrics) else base_target_metrics.get("acc"),
            "robust_target_acc": None if not (target_has_label and robust_target_metrics) else robust_target_metrics.get("acc"),
            "base_target_weighted_f1": None if not (target_has_label and base_target_metrics) else base_target_metrics.get("weighted_f1"),
            "robust_target_weighted_f1": None if not (target_has_label and robust_target_metrics) else robust_target_metrics.get("weighted_f1"),
            "base_top2": None if not (target_has_label and base_target_metrics) else base_target_metrics.get("top2"),
            "robust_top2": None if not (target_has_label and robust_target_metrics) else robust_target_metrics.get("top2"),
            "base_top3": None if not (target_has_label and base_target_metrics) else base_target_metrics.get("top3"),
            "robust_top3": None if not (target_has_label and robust_target_metrics) else robust_target_metrics.get("top3"),
            "target_pred_stats_path": stats_path,
            "target_pred_counts_path": counts_csv_path,
            "target_pred_csv_path": pred_csv_path,
        }
        with open(os.path.join(args.output_dir, f"cct_summary_{view}.json"), "w") as f:
            json.dump(cct_summary_view, f, indent=2, sort_keys=True)

        # Save robust model under new model_ts (run_id + suffix).
        if "/" not in model_ts:
            raise ValueError(f"model_ts must be like '<output>/<run_id>', got: {model_ts}")
        output, run_id = model_ts.split("/", 1)
        new_model_ts = f"{output}/{run_id}{args.output_suffix}"
        new_root = os.path.join("model-classifier", new_model_ts, model_name)
        ensure_dir(new_root)

        robust_info = dict(info)
        robust_info["timestamp"] = datetime.now().strftime("%Y%m%d%H%M")
        robust_info["cct_robust_classifier"] = {
            "enabled": True,
            "source_dataset": args.source_dataset,
            "target_csv": args.target_csv,
            "view": view,
            "steps": args.steps,
            "lr": args.lr,
            "shift_weight": args.shift_weight,
            "anchor_weight": args.anchor_weight,
            "cct_shift_rho": args.cct_shift_rho,
            "cct_target_cluster_ratio": args.cct_target_cluster_ratio,
            "cct_assignment_tau": args.cct_assignment_tau,
            "cct_dp_lambda": args.cct_dp_lambda,
            "cct_dp_percentile": args.cct_dp_percentile,
            "cct_distance": args.cct_distance,
            "cct_min_proto_mass": args.cct_min_proto_mass,
            "target_cluster_count": int(K),
            "source_proto_count_min": cct_summary_view["source_proto_count_min"],
            "source_proto_count_mean": cct_summary_view["source_proto_count_mean"],
            "source_proto_count_max": cct_summary_view["source_proto_count_max"],
            "delta_norm_min": cct_summary_view["delta_norm_min"],
            "delta_norm_mean": cct_summary_view["delta_norm_mean"],
            "delta_norm_max": cct_summary_view["delta_norm_max"],
            "assignment_entropy_min": cct_summary_view["assignment_entropy_min"],
            "assignment_entropy_mean": cct_summary_view["assignment_entropy_mean"],
            "assignment_entropy_max": cct_summary_view["assignment_entropy_max"],
            "best_step": int(best["step"]),
            "best_val_weighted_f1": float(best["metrics"]["weighted_f1"]) if best["metrics"] else 0.0,
            "base_target_acc": cct_summary_view["base_target_acc"],
            "robust_target_acc": cct_summary_view["robust_target_acc"],
            "base_target_weighted_f1": cct_summary_view["base_target_weighted_f1"],
            "robust_target_weighted_f1": cct_summary_view["robust_target_weighted_f1"],
            "base_top2": cct_summary_view["base_top2"],
            "robust_top2": cct_summary_view["robust_top2"],
            "base_top3": cct_summary_view["base_top3"],
            "robust_top3": cct_summary_view["robust_top3"],
            "diagnostic_output_dir": args.output_dir,
            "cct_summary_path": os.path.join(args.output_dir, f"cct_summary_{view}.json"),
            "cct_delta_path": view_prefix + "_delta.npy",
            "cct_assignment_path": view_prefix + "_assignment.npy",
            "cct_target_cluster_centers_path": view_prefix + "_target_cluster_centers.npy",
            "cct_target_cluster_masses_path": view_prefix + "_target_cluster_masses.npy",
            "cct_source_prototypes_path": view_prefix + "_source_prototypes.npy",
            "cct_source_prototype_class_ids_path": view_prefix + "_source_prototype_class_ids.npy",
            "cct_source_prototype_masses_path": view_prefix + "_source_prototype_masses.npy",
            "target_pred_stats_path": stats_path,
            "target_pred_counts_path": counts_csv_path,
            "target_pred_csv_path": pred_csv_path,
        }
        with open(os.path.join(new_root, "info.json"), "w") as f:
            json.dump(robust_info, f, indent=2, sort_keys=True)

        new_model_dir = model_dir_for_view(new_root, view)
        ensure_dir(new_model_dir)
        model = model.to("cpu")
        torch.save(model.state_dict(), os.path.join(new_model_dir, "pytorch_model.bin"))

        summary.setdefault("cct_robust_models", {})[view] = {
            "base_model_ts": model_ts,
            "robust_model_ts": new_model_ts,
            "best_step": int(best["step"]),
            "base_val_weighted_f1": float(base_val_weighted_f1),
            "best_val_weighted_f1": float(best["metrics"]["weighted_f1"]) if best["metrics"] else 0.0,
            "target_cluster_count": int(K),
            "source_proto_count_min": cct_summary_view["source_proto_count_min"],
            "source_proto_count_mean": cct_summary_view["source_proto_count_mean"],
            "source_proto_count_max": cct_summary_view["source_proto_count_max"],
            **delta_norm_stats,
            **assignment_entropy_stats,
        }

    out_path = os.path.join("model-classifier", f"cct_robust_classifier_summary_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[CCT] Summary saved:", out_path)


if __name__ == "__main__":
    main()
