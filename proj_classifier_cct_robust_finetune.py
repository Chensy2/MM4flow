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
                if len(idx) == 0:
                    continue
                new_protos.append(np.mean(features[idx], axis=0).astype(np.float32))
            if not new_protos:
                proto_arr = center.astype(np.float32)[None, :]
                masses = np.array([features.shape[0]], dtype=np.int64)
            else:
                proto_arr = np.stack(new_protos, axis=0)
                if metric == "euclidean":
                    dists = np.linalg.norm(features[:, None, :] - proto_arr[None, :, :], axis=2)
                else:
                    fn = _l2_normalize(features.astype(np.float64))
                    pn = _l2_normalize(proto_arr.astype(np.float64))
                    dists = 1.0 - (fn @ pn.T)
                assign = np.argmin(dists, axis=1).astype(np.int64)
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


def point_to_center_distance(points, center, metric):
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if points.size == 0:
        return np.asarray([], dtype=np.float64)
    if metric == "euclidean":
        return np.linalg.norm(points - center[None, :], axis=1)
    points_n = _l2_normalize(points)
    center_norm = max(1e-12, np.linalg.norm(center))
    center_n = center / center_norm
    return 1.0 - (points_n @ center_n.reshape(-1, 1)).reshape(-1)


def target_cluster_intra_distances(features, cluster_ids, centers, metric):
    features = np.asarray(features, dtype=np.float32)
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
    centers = np.asarray(centers, dtype=np.float32)
    K = centers.shape[0]
    intra = np.zeros((K,), dtype=np.float64)
    for k in range(K):
        idx = np.where(cluster_ids == k)[0]
        if len(idx) == 0:
            intra[k] = 0.0
        else:
            intra[k] = float(np.mean(point_to_center_distance(features[idx], centers[k], metric)))
    return intra.astype(np.float64)


def class_weighted_cluster_stat(A, target_masses, cluster_values, eps=1e-8, orphan_value=None):
    A = np.asarray(A, dtype=np.float64)
    target_masses = np.asarray(target_masses, dtype=np.float64)
    cluster_values = np.asarray(cluster_values, dtype=np.float64)
    weighted = A * target_masses[:, None]
    class_mass = np.sum(weighted, axis=0)
    values = (weighted.T @ cluster_values) / np.maximum(class_mass, float(eps))
    orphan = class_mass <= float(eps)
    if orphan_value is not None:
        values[orphan] = float(orphan_value)
    return values.astype(np.float64), class_mass.astype(np.float64), orphan.astype(bool)


def normalize_positive(values, orphan=None, eps=1e-8):
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values)
    if orphan is not None:
        mask &= ~np.asarray(orphan, dtype=bool)
    if not np.any(mask):
        return np.ones_like(values, dtype=np.float64)
    scale = float(np.percentile(values[mask], 90))
    if scale < float(eps):
        scale = float(np.max(values[mask]))
    if scale < float(eps):
        scale = 1.0
    out = values / scale
    out = np.clip(out, 0.0, 10.0)
    if orphan is not None:
        out[np.asarray(orphan, dtype=bool)] = 1.0
    return out.astype(np.float64)


def class_shift_reliability(entropy_norm, intra_distance_norm, orphan=None):
    entropy_norm = np.asarray(entropy_norm, dtype=np.float64)
    intra_distance_norm = np.asarray(intra_distance_norm, dtype=np.float64)
    reliability = np.exp(-entropy_norm) * np.exp(-intra_distance_norm)
    if orphan is not None:
        reliability[np.asarray(orphan, dtype=bool)] = 0.0
    return reliability.astype(np.float64)


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or y.size < 2:
        return None
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_average(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty_like(values, dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = np.mean(np.arange(start, end, dtype=np.float64)) + 1.0
        start = end
    return ranks


def safe_spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or y.size < 2:
        return None
    return safe_corr(rankdata_average(x), rankdata_average(y))


def class_assignment_entropy(A, cluster_entropy, target_masses, eps=1e-8):
    A = np.asarray(A, dtype=np.float64)
    cluster_entropy = np.asarray(cluster_entropy, dtype=np.float64)
    target_masses = np.asarray(target_masses, dtype=np.float64)
    weighted = A * target_masses[:, None]
    class_mass = np.sum(weighted, axis=0)
    orphan = class_mass <= float(eps)
    entropy_by_class = (weighted.T @ cluster_entropy) / np.maximum(class_mass, float(eps))
    max_entropy = np.log(max(2, A.shape[1]))
    entropy_by_class[orphan] = max_entropy
    entropy_norm = entropy_by_class / max(float(eps), max_entropy)
    entropy_norm[orphan] = 1.0
    return (
        entropy_by_class.astype(np.float64),
        entropy_norm.astype(np.float64),
        class_mass.astype(np.float64),
        orphan.astype(bool),
    )


def prediction_im_metrics(probs):
    probs = np.asarray(probs, dtype=np.float64)
    mean_prob = np.mean(probs, axis=0)
    h_global = -float(np.sum(mean_prob * np.log(mean_prob + 1e-12)))
    h_local = float(np.mean(probs_entropy(probs)))
    return {
        "global_entropy": h_global,
        "local_entropy": h_local,
        "im_score": h_global - h_local,
    }


def plot_target_eval_metrics(history_rows, output_dir, view):
    if not history_rows:
        return None, None
    df = pd.DataFrame(history_rows)
    csv_path = os.path.join(output_dir, f"cct_target_eval_history_{view}.csv")
    json_path = os.path.join(output_dir, f"cct_target_eval_metric_correlations_{view}.json")
    plot_path = os.path.join(output_dir, f"cct_target_eval_metrics_vs_wf1_{view}.png")
    df.to_csv(csv_path, index=False)

    corr = {"view": view, "csv_path": csv_path, "plot_path": plot_path}
    has_wf1 = "target_weighted_f1" in df.columns and df["target_weighted_f1"].notna().any()
    metrics = ["global_entropy", "local_entropy", "im_score"]
    if has_wf1:
        y = df["target_weighted_f1"].values.astype(np.float64)
        for metric in metrics:
            x = df[metric].values.astype(np.float64)
            corr[f"pearson_{metric}_target_wf1"] = safe_corr(x, y)
            corr[f"spearman_{metric}_target_wf1"] = safe_spearman(x, y)
    else:
        for metric in metrics:
            corr[f"pearson_{metric}_target_wf1"] = None
            corr[f"spearman_{metric}_target_wf1"] = None
        plot_path = None
        corr["plot_path"] = None

    plot_error = None
    if has_wf1:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            labels = {
                "global_entropy": "Global Entropy",
                "local_entropy": "Local Entropy",
                "im_score": "IM Score",
            }
            for ax, metric in zip(axes, metrics):
                ax.scatter(df[metric], df["target_weighted_f1"], alpha=0.8)
                ax.plot(df[metric], df["target_weighted_f1"], alpha=0.45)
                pearson = corr[f"pearson_{metric}_target_wf1"]
                title = labels[metric]
                if pearson is not None:
                    title += f" r={pearson:.3f}"
                ax.set_title(title)
                ax.set_xlabel(labels[metric])
                ax.set_ylabel("Target WF1")
            fig.suptitle(f"{view}: target probability metrics vs Target WF1")
            fig.tight_layout()
            fig.savefig(plot_path, dpi=180)
            plt.close(fig)
        except Exception as exc:
            plot_error = str(exc)
            plot_path = None
            corr["plot_path"] = None
    corr["plot_error"] = plot_error
    with open(json_path, "w") as f:
        json.dump(corr, f, indent=2, sort_keys=True)
    corr["json_path"] = json_path
    return csv_path, corr


def write_class_delta_plot(df, output_dir, view, suffix, title):
    csv_path = os.path.join(output_dir, f"cct_entropy_delta_f1_{suffix}_{view}.csv")
    json_path = os.path.join(output_dir, f"cct_entropy_delta_f1_{suffix}_{view}.json")
    plot_path = os.path.join(output_dir, f"cct_entropy_delta_f1_{suffix}_{view}.png")
    df.to_csv(csv_path, index=False)

    valid = (df["support"].values > 0) & np.isfinite(df["target_cluster_assignment_entropy"].values)
    valid &= np.isfinite(df["delta_f1_from_step0"].values)
    entropy_x = df.loc[valid, "target_cluster_assignment_entropy"].values
    delta_y = df.loc[valid, "delta_f1_from_step0"].values
    pearson = safe_corr(entropy_x, delta_y)
    spearman = safe_spearman(entropy_x, delta_y)

    plot_error = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 5))
        sizes = np.clip(df.loc[valid, "support"].values.astype(np.float64), 1.0, None)
        sizes = 25.0 + 125.0 * sizes / max(1.0, np.max(sizes))
        plt.scatter(entropy_x, delta_y, s=sizes, alpha=0.75, edgecolors="black", linewidths=0.4)
        plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        plot_title = title
        if pearson is not None:
            plot_title += f" | r={pearson:.3f}"
        if spearman is not None:
            plot_title += f", rho={spearman:.3f}"
        plt.title(plot_title)
        plt.xlabel("Class target-cluster assignment entropy")
        plt.ylabel("Delta F1 from step 0")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()
    except Exception as exc:
        plot_error = str(exc)
        plot_path = None

    analysis = {
        "view": view,
        "suffix": suffix,
        "num_classes": int(len(df)),
        "num_classes_with_support": int(np.sum(valid)),
        "pearson_entropy_delta_f1": pearson,
        "spearman_entropy_delta_f1": spearman,
        "mean_delta_f1": float(np.mean(delta_y)) if delta_y.size else None,
        "median_delta_f1": float(np.median(delta_y)) if delta_y.size else None,
        "mean_entropy": float(np.mean(entropy_x)) if entropy_x.size else None,
        "median_entropy": float(np.median(entropy_x)) if entropy_x.size else None,
        "csv_path": csv_path,
        "plot_path": plot_path,
        "plot_error": plot_error,
    }
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2, sort_keys=True)
    analysis["json_path"] = json_path
    return analysis


def write_reliability_delta_plots(df, output_dir, view, suffix, title):
    csv_path = os.path.join(output_dir, f"cct_reliability_delta_f1_{suffix}_{view}.csv")
    json_path = os.path.join(output_dir, f"cct_reliability_delta_f1_{suffix}_{view}.json")
    plot_path = os.path.join(output_dir, f"cct_reliability_delta_f1_{suffix}_{view}.png")
    df.to_csv(csv_path, index=False)

    metric_specs = [
        ("target_cluster_assignment_entropy", "Assignment Entropy"),
        ("target_cluster_intra_distance", "Target Intra-Cluster Distance"),
        ("shift_reliability", "Shift Reliability"),
    ]
    analysis = {"view": view, "suffix": suffix, "csv_path": csv_path, "plot_path": plot_path}
    for key, _ in metric_specs:
        valid = (df["support"].values > 0) & np.isfinite(df[key].values)
        valid &= np.isfinite(df["delta_f1_from_step0"].values)
        x = df.loc[valid, key].values
        y = df.loc[valid, "delta_f1_from_step0"].values
        analysis[f"pearson_{key}_delta_f1"] = safe_corr(x, y)
        analysis[f"spearman_{key}_delta_f1"] = safe_spearman(x, y)

    plot_error = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (key, label) in zip(axes, metric_specs):
            valid = (df["support"].values > 0) & np.isfinite(df[key].values)
            valid &= np.isfinite(df["delta_f1_from_step0"].values)
            sizes = np.clip(df.loc[valid, "support"].values.astype(np.float64), 1.0, None)
            sizes = 20.0 + 100.0 * sizes / max(1.0, np.max(sizes))
            ax.scatter(
                df.loc[valid, key].values,
                df.loc[valid, "delta_f1_from_step0"].values,
                s=sizes,
                alpha=0.75,
                edgecolors="black",
                linewidths=0.35,
            )
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
            pearson = analysis[f"pearson_{key}_delta_f1"]
            subtitle = label
            if pearson is not None:
                subtitle += f" r={pearson:.3f}"
            ax.set_title(subtitle)
            ax.set_xlabel(label)
            ax.set_ylabel("Delta F1 from step 0")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
    except Exception as exc:
        plot_error = str(exc)
        plot_path = None
        analysis["plot_path"] = None
    analysis["plot_error"] = plot_error
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2, sort_keys=True)
    analysis["json_path"] = json_path
    return analysis


def write_entropy_delta_f1_diagnostics(
    *,
    output_dir,
    view,
    idx2label,
    target_labels,
    base_preds,
    robust_preds,
    A,
    cluster_entropy,
    target_masses,
    delta_norms,
    base_counts,
    base_fracs,
    robust_counts,
    robust_fracs,
    class_intra_distance=None,
    class_intra_distance_norm=None,
    shift_reliability=None,
):
    y_true = target_labels.detach().cpu().numpy().astype(np.int64)
    num_classes = len(idx2label)
    labels = np.arange(num_classes, dtype=np.int64)
    base_p, base_r, base_f1, support = precision_recall_fscore_support(
        y_true, base_preds, labels=labels, average=None, zero_division=0
    )
    robust_p, robust_r, robust_f1, _ = precision_recall_fscore_support(
        y_true, robust_preds, labels=labels, average=None, zero_division=0
    )
    cls_entropy, cls_entropy_norm, cls_mass, orphan = class_assignment_entropy(A, cluster_entropy, target_masses)
    if class_intra_distance is None:
        class_intra_distance = np.zeros((num_classes,), dtype=np.float64)
    if class_intra_distance_norm is None:
        class_intra_distance_norm = np.zeros((num_classes,), dtype=np.float64)
    if shift_reliability is None:
        shift_reliability = class_shift_reliability(cls_entropy_norm, class_intra_distance_norm, orphan)
    delta_f1 = robust_f1 - base_f1

    rows = []
    for i in range(num_classes):
        rows.append(
            {
                "class_idx": int(i),
                "class": idx2label[i],
                "support": int(support[i]),
                "f1_before_step0": float(base_f1[i]),
                "f1_after": float(robust_f1[i]),
                "delta_f1_from_step0": float(delta_f1[i]),
                "f1_before": float(base_f1[i]),
                "delta_f1": float(delta_f1[i]),
                "precision_before": float(base_p[i]),
                "precision_after": float(robust_p[i]),
                "recall_before": float(base_r[i]),
                "recall_after": float(robust_r[i]),
                "target_cluster_assignment_entropy": float(cls_entropy[i]),
                "target_cluster_assignment_entropy_norm": float(cls_entropy_norm[i]),
                "target_cluster_soft_mass": float(cls_mass[i]),
                "is_orphan_class": bool(orphan[i]),
                "target_cluster_intra_distance": float(class_intra_distance[i]),
                "target_cluster_intra_distance_norm": float(class_intra_distance_norm[i]),
                "shift_reliability": float(shift_reliability[i]),
                "delta_norm": float(delta_norms[i]),
                "base_pred_count": int(base_counts[i]),
                "robust_pred_count": int(robust_counts[i]),
                "base_pred_frac": float(base_fracs[i]),
                "robust_pred_frac": float(robust_fracs[i]),
            }
        )

    df = pd.DataFrame(rows)
    analysis = write_class_delta_plot(df, output_dir, view, "final", f"{view}: final target cluster entropy vs Delta F1")
    reliability_analysis = write_reliability_delta_plots(
        df, output_dir, view, "final", f"{view}: final reliability diagnostics vs Delta F1"
    )
    analysis["reliability_analysis"] = reliability_analysis
    return analysis


def pseudo_metric_on_mask(preds, pseudo_labels, mask, num_classes):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return {"n": 0, "acc": None, "weighted_f1": None}
    y = np.asarray(pseudo_labels, dtype=np.int64)[mask]
    p = np.asarray(preds, dtype=np.int64)[mask]
    acc = float(np.mean(p == y))
    _, _, f1, _ = precision_recall_fscore_support(
        y, p, labels=np.arange(num_classes, dtype=np.int64), average="weighted", zero_division=0
    )
    return {"n": int(np.sum(mask)), "acc": acc, "weighted_f1": float(f1)}


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def evaluate_probs_metrics(probs, labels=None, report_topk=3):
    probs = np.asarray(probs, dtype=np.float64)
    preds = np.argmax(probs, axis=1).astype(np.int64)
    out = {"preds": preds}
    if labels is None:
        return out
    y_true = np.asarray(labels, dtype=np.int64)
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


def mix_probs_by_class_gate(base_probs, robust_probs, gate):
    base_probs = np.asarray(base_probs, dtype=np.float64)
    robust_probs = np.asarray(robust_probs, dtype=np.float64)
    gate = np.asarray(gate, dtype=np.float64).reshape(1, -1)
    mixed = gate * robust_probs + (1.0 - gate) * base_probs
    mixed = mixed / np.maximum(np.sum(mixed, axis=1, keepdims=True), 1e-12)
    return mixed


def write_pseudo_agreement_audit(output_dir, idx2label, pseudo_cache, target_labels=None):
    views = [v for v in ["ps", "byte", "mm"] if v in pseudo_cache]
    if len(views) < 2:
        return None

    lengths = [len(pseudo_cache[v]["base_preds"]) for v in views]
    n = min(lengths)
    if len(set(lengths)) != 1:
        print(f"[CCT][PseudoAudit] Warning: target prediction lengths differ {dict(zip(views, lengths))}; using first {n}.")
    num_classes = len(idx2label)
    base_preds = {v: np.asarray(pseudo_cache[v]["base_preds"][:n], dtype=np.int64) for v in views}
    robust_preds = {v: np.asarray(pseudo_cache[v]["robust_preds"][:n], dtype=np.int64) for v in views}
    y_true = None
    if target_labels is not None:
        y_true = np.asarray(target_labels[:n], dtype=np.int64)

    groups = []
    if len(views) >= 3 and all(v in base_preds for v in ["ps", "byte", "mm"]):
        ps, byte, mm = base_preds["ps"], base_preds["byte"], base_preds["mm"]
        mask3 = (ps == byte) & (ps == mm)
        groups.append(("agree3", "ps+byte+mm", "", mask3, ps))
        pairs = [
            ("agree2_ps_byte", "ps+byte", "mm", (ps == byte) & (ps != mm), ps),
            ("agree2_ps_mm", "ps+mm", "byte", (ps == mm) & (ps != byte), ps),
            ("agree2_byte_mm", "byte+mm", "ps", (byte == mm) & (byte != ps), byte),
        ]
        groups.extend(pairs)
    else:
        for i in range(len(views)):
            for j in range(i + 1, len(views)):
                v1, v2 = views[i], views[j]
                mask = base_preds[v1] == base_preds[v2]
                groups.append((f"agree2_{v1}_{v2}", f"{v1}+{v2}", "", mask, base_preds[v1]))

    summary_rows = []
    class_rows = []
    view_rows = []
    sample_frames = []
    for group_name, source_views, heldout_view, mask, pseudo_labels in groups:
        mask = np.asarray(mask, dtype=bool)
        selected = int(np.sum(mask))
        if selected == 0:
            pseudo_acc = None
        elif y_true is None:
            pseudo_acc = None
        else:
            pseudo_acc = float(np.mean(pseudo_labels[mask] == y_true[mask]))
        counts = np.bincount(pseudo_labels[mask], minlength=num_classes).astype(np.int64) if selected else np.zeros(num_classes, dtype=np.int64)
        summary_rows.append(
            {
                "group": group_name,
                "source_views": source_views,
                "heldout_view": heldout_view,
                "n": selected,
                "coverage": float(selected / max(1, n)),
                "pseudo_acc_if_labeled": pseudo_acc,
                "num_pseudo_classes": int(np.sum(counts > 0)),
                "max_class_frac": float(np.max(counts) / max(1, selected)) if selected else 0.0,
            }
        )

        if selected:
            df_sample = pd.DataFrame(
                {
                    "sample_idx": np.where(mask)[0].astype(np.int64),
                    "group": group_name,
                    "source_views": source_views,
                    "heldout_view": heldout_view,
                    "pseudo_label_idx": pseudo_labels[mask].astype(np.int64),
                    "pseudo_label": [idx2label[int(i)] for i in pseudo_labels[mask]],
                }
            )
            if y_true is not None:
                df_sample["target_label_idx"] = y_true[mask].astype(np.int64)
                df_sample["target_label"] = [idx2label[int(i)] for i in y_true[mask]]
                df_sample["pseudo_correct"] = df_sample["pseudo_label_idx"].values == df_sample["target_label_idx"].values
            for v in views:
                df_sample[f"{v}_base_pred_idx"] = base_preds[v][mask]
                df_sample[f"{v}_base_pred"] = [idx2label[int(i)] for i in base_preds[v][mask]]
                df_sample[f"{v}_robust_pred_idx"] = robust_preds[v][mask]
                df_sample[f"{v}_robust_pred"] = [idx2label[int(i)] for i in robust_preds[v][mask]]
            sample_frames.append(df_sample)

        for v in views:
            base_metric = pseudo_metric_on_mask(base_preds[v], pseudo_labels, mask, num_classes)
            robust_metric = pseudo_metric_on_mask(robust_preds[v], pseudo_labels, mask, num_classes)
            view_rows.append(
                {
                    "group": group_name,
                    "view": v,
                    "source_views": source_views,
                    "heldout_view": heldout_view,
                    "is_heldout_view": bool(v == heldout_view),
                    "n": selected,
                    "base_acc_vs_pseudo": base_metric["acc"],
                    "robust_acc_vs_pseudo": robust_metric["acc"],
                    "delta_acc_vs_pseudo": None
                    if base_metric["acc"] is None or robust_metric["acc"] is None
                    else float(robust_metric["acc"] - base_metric["acc"]),
                    "base_weighted_f1_vs_pseudo": base_metric["weighted_f1"],
                    "robust_weighted_f1_vs_pseudo": robust_metric["weighted_f1"],
                    "delta_weighted_f1_vs_pseudo": None
                    if base_metric["weighted_f1"] is None or robust_metric["weighted_f1"] is None
                    else float(robust_metric["weighted_f1"] - base_metric["weighted_f1"]),
                }
            )

        for c in range(num_classes):
            cls_mask = mask & (pseudo_labels == c)
            cls_n = int(np.sum(cls_mask))
            if cls_n == 0:
                continue
            row = {
                "group": group_name,
                "source_views": source_views,
                "heldout_view": heldout_view,
                "class_idx": int(c),
                "class": idx2label[c],
                "pseudo_support": cls_n,
                "pseudo_support_frac_in_group": float(cls_n / max(1, selected)),
                "pseudo_acc_if_labeled": None if y_true is None else float(np.mean(y_true[cls_mask] == c)),
            }
            for v in views:
                base_recall = float(np.mean(base_preds[v][cls_mask] == c))
                robust_recall = float(np.mean(robust_preds[v][cls_mask] == c))
                row[f"{v}_base_recall_vs_pseudo"] = base_recall
                row[f"{v}_robust_recall_vs_pseudo"] = robust_recall
                row[f"{v}_delta_recall_vs_pseudo"] = robust_recall - base_recall
            class_rows.append(row)

    summary_path = os.path.join(output_dir, "cct_pseudo_agreement_summary.json")
    view_metrics_path = os.path.join(output_dir, "cct_pseudo_agreement_view_metrics.csv")
    class_summary_path = os.path.join(output_dir, "cct_pseudo_agreement_class_summary.csv")
    sample_path = os.path.join(output_dir, "cct_pseudo_agreement_samples.csv.gz")
    pd.DataFrame(view_rows).to_csv(view_metrics_path, index=False)
    pd.DataFrame(class_rows).to_csv(class_summary_path, index=False)
    if sample_frames:
        pd.concat(sample_frames, ignore_index=True).to_csv(sample_path, index=False, compression="gzip")
    else:
        sample_path = None
    summary_payload = {
        "views": views,
        "num_samples": int(n),
        "target_has_label": bool(y_true is not None),
        "groups": summary_rows,
        "view_metrics_path": view_metrics_path,
        "class_summary_path": class_summary_path,
        "sample_path": sample_path,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2, sort_keys=True)
    summary_payload["summary_path"] = summary_path
    return summary_payload


def write_pseudo_gate_audit(
    output_dir,
    idx2label,
    pseudo_cache,
    target_labels=None,
    *,
    support_prior=50.0,
    min_support=10,
    tau=0.05,
    report_topk=3,
):
    views = [v for v in ["ps", "byte", "mm"] if v in pseudo_cache]
    if len(views) < 2:
        return None
    lengths = [len(pseudo_cache[v]["base_preds"]) for v in views]
    n = min(lengths)
    num_classes = len(idx2label)
    labels = np.arange(num_classes, dtype=np.int64)
    base_preds = {v: np.asarray(pseudo_cache[v]["base_preds"][:n], dtype=np.int64) for v in views}
    robust_preds = {v: np.asarray(pseudo_cache[v]["robust_preds"][:n], dtype=np.int64) for v in views}
    base_probs = {v: np.asarray(pseudo_cache[v]["base_probs"][:n], dtype=np.float64) for v in views}
    robust_probs = {v: np.asarray(pseudo_cache[v]["robust_probs"][:n], dtype=np.float64) for v in views}
    y_true = None if target_labels is None else np.asarray(target_labels[:n], dtype=np.int64)

    agree3_mask = None
    agree3_labels = None
    heldout_specs = {}
    if all(v in base_preds for v in ["ps", "byte", "mm"]):
        ps, byte, mm = base_preds["ps"], base_preds["byte"], base_preds["mm"]
        agree3_mask = (ps == byte) & (ps == mm)
        agree3_labels = ps
        heldout_specs = {
            "ps": ("agree2_byte_mm", (byte == mm) & (byte != ps), byte),
            "byte": ("agree2_ps_mm", (ps == mm) & (ps != byte), ps),
            "mm": ("agree2_ps_byte", (ps == byte) & (ps != mm), ps),
        }

    gate_rows = []
    metric_rows = []
    for view in views:
        gate_agree3 = np.zeros((num_classes,), dtype=np.float64)
        gate_heldout = np.zeros((num_classes,), dtype=np.float64)
        gate_combined = np.zeros((num_classes,), dtype=np.float64)
        for c in range(num_classes):
            row = {"view": view, "class_idx": int(c), "class": idx2label[c]}
            if agree3_mask is not None:
                cls_mask = agree3_mask & (agree3_labels == c)
                n3 = int(np.sum(cls_mask))
                support_conf3 = float(n3 / (n3 + float(support_prior))) if n3 > 0 else 0.0
                base_r3 = float(np.mean(base_preds[view][cls_mask] == c)) if n3 else 0.0
                robust_r3 = float(np.mean(robust_preds[view][cls_mask] == c)) if n3 else 0.0
                delta3 = robust_r3 - base_r3
                # agree3 is generated by base predictions, so it mostly acts as a preservation signal.
                gate3 = 0.0 if n3 < int(min_support) else support_conf3 * sigmoid_np(delta3 / float(tau))[()]
                gate_agree3[c] = float(gate3)
                row.update(
                    {
                        "agree3_support": n3,
                        "agree3_support_conf": support_conf3,
                        "agree3_base_recall_vs_pseudo": base_r3,
                        "agree3_robust_recall_vs_pseudo": robust_r3,
                        "agree3_delta_recall_vs_pseudo": delta3,
                        "agree3_gate": float(gate3),
                    }
                )
            if view in heldout_specs:
                group, mask2, labels2 = heldout_specs[view]
                cls_mask = mask2 & (labels2 == c)
                n2 = int(np.sum(cls_mask))
                support_conf2 = float(n2 / (n2 + float(support_prior))) if n2 > 0 else 0.0
                base_r2 = float(np.mean(base_preds[view][cls_mask] == c)) if n2 else 0.0
                robust_r2 = float(np.mean(robust_preds[view][cls_mask] == c)) if n2 else 0.0
                delta2 = robust_r2 - base_r2
                gate2 = 0.0 if n2 < int(min_support) else support_conf2 * sigmoid_np(delta2 / float(tau))[()]
                gate_heldout[c] = float(gate2)
                row.update(
                    {
                        "heldout_group": group,
                        "heldout_support": n2,
                        "heldout_support_conf": support_conf2,
                        "heldout_base_recall_vs_pseudo": base_r2,
                        "heldout_robust_recall_vs_pseudo": robust_r2,
                        "heldout_delta_recall_vs_pseudo": delta2,
                        "heldout_gate": float(gate2),
                    }
                )
            combined = gate_heldout[c] if gate_heldout[c] > 0 else gate_agree3[c]
            gate_combined[c] = float(combined)
            row["combined_gate"] = float(combined)
            if combined <= 1e-8:
                row["gate_reason"] = "base_default_or_low_support"
            elif gate_heldout[c] > 0:
                row["gate_reason"] = "heldout_agree2_supports_robust"
            else:
                row["gate_reason"] = "agree3_preservation_mix"
            gate_rows.append(row)

        eval_labels = y_true
        base_metrics = evaluate_probs_metrics(base_probs[view], eval_labels, report_topk=report_topk)
        robust_metrics = evaluate_probs_metrics(robust_probs[view], eval_labels, report_topk=report_topk)
        for name, gate in [
            ("agree3_gate", gate_agree3),
            ("heldout_gate", gate_heldout),
            ("combined_gate", gate_combined),
        ]:
            mixed = mix_probs_by_class_gate(base_probs[view], robust_probs[view], gate)
            metrics = evaluate_probs_metrics(mixed, eval_labels, report_topk=report_topk)
            metric_rows.append(
                {
                    "view": view,
                    "gate_type": name,
                    "mean_gate": float(np.mean(gate)),
                    "num_classes_gate_gt_0": int(np.sum(gate > 1e-8)),
                    "num_classes_gate_gt_0_5": int(np.sum(gate > 0.5)),
                    "base_acc": base_metrics.get("acc"),
                    "robust_acc": robust_metrics.get("acc"),
                    "gated_acc": metrics.get("acc"),
                    "base_weighted_f1": base_metrics.get("weighted_f1"),
                    "robust_weighted_f1": robust_metrics.get("weighted_f1"),
                    "gated_weighted_f1": metrics.get("weighted_f1"),
                    "base_top2": base_metrics.get("top2"),
                    "robust_top2": robust_metrics.get("top2"),
                    "gated_top2": metrics.get("top2"),
                    "base_top3": base_metrics.get("top3"),
                    "robust_top3": robust_metrics.get("top3"),
                    "gated_top3": metrics.get("top3"),
                }
            )

    gate_path = os.path.join(output_dir, "cct_pseudo_base_robust_gate_by_class.csv")
    metrics_path = os.path.join(output_dir, "cct_pseudo_base_robust_gate_metrics.csv")
    summary_path = os.path.join(output_dir, "cct_pseudo_base_robust_gate_summary.json")
    pd.DataFrame(gate_rows).to_csv(gate_path, index=False)
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    payload = {
        "views": views,
        "num_samples": int(n),
        "target_has_label": bool(y_true is not None),
        "support_prior": float(support_prior),
        "min_support": int(min_support),
        "tau": float(tau),
        "gate_by_class_path": gate_path,
        "gate_metrics_path": metrics_path,
        "metrics": metric_rows,
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    payload["summary_path"] = summary_path
    return payload


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
    eval_callback=None,
):
    head = head.to(device)
    head.train()
    head_orig_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
    opt = torch.optim.Adam(head.parameters(), lr=lr, eps=1e-8)

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
    if eval_callback is not None:
        eval_callback(0, head)

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
            if eval_callback is not None:
                eval_callback(step, head)
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
    pseudo_audit_cache = {}
    pseudo_audit_target_labels = None

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
        target_cluster_intra = target_cluster_intra_distances(
            target_np, target_ids, target_centers, metric=args.cct_distance
        )

        # Step 4/5: alignment and delta
        A, ent, mu_target = compute_alignment(
            target_centers.astype(np.float32),
            target_masses.astype(np.float32),
            source_protos_by_class,
            metric=args.cct_distance,
            tau=args.cct_assignment_tau,
        )
        delta = (mu_target - source_mu).astype(np.float32)  # [C,D]
        class_entropy_all, class_entropy_norm_all, class_soft_mass_all, class_orphan_all = class_assignment_entropy(
            A, ent, target_masses
        )
        max_intra = float(np.max(target_cluster_intra)) if target_cluster_intra.size else 0.0
        class_intra_distance, _, _ = class_weighted_cluster_stat(
            A, target_masses, target_cluster_intra, orphan_value=max_intra
        )
        class_intra_distance_norm = normalize_positive(class_intra_distance, orphan=class_orphan_all)
        class_reliability = class_shift_reliability(
            class_entropy_norm_all, class_intra_distance_norm, orphan=class_orphan_all
        )

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
        intra_distance_stats = {
            "target_cluster_intra_distance_min": float(np.min(target_cluster_intra)) if target_cluster_intra.size else 0.0,
            "target_cluster_intra_distance_mean": float(np.mean(target_cluster_intra)) if target_cluster_intra.size else 0.0,
            "target_cluster_intra_distance_max": float(np.max(target_cluster_intra)) if target_cluster_intra.size else 0.0,
            "class_target_intra_distance_min": float(np.min(class_intra_distance)) if class_intra_distance.size else 0.0,
            "class_target_intra_distance_mean": float(np.mean(class_intra_distance)) if class_intra_distance.size else 0.0,
            "class_target_intra_distance_max": float(np.max(class_intra_distance)) if class_intra_distance.size else 0.0,
            "class_shift_reliability_min": float(np.min(class_reliability)) if class_reliability.size else 0.0,
            "class_shift_reliability_mean": float(np.mean(class_reliability)) if class_reliability.size else 0.0,
            "class_shift_reliability_max": float(np.max(class_reliability)) if class_reliability.size else 0.0,
        }

        # Step 6: head-only finetune
        for _, p in model.named_parameters():
            p.requires_grad_(False)
        head = classifier_module(model, view)
        for p in head.parameters():
            p.requires_grad_(True)

        base_head_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
        delta_torch = torch.from_numpy(delta)
        head = head.to(device)
        entropy_delta_f1_history = []
        entropy_delta_f1_history_path = None
        entropy_delta_f1_correlations = []
        entropy_delta_f1_correlations_path = None
        target_eval_history = []
        target_eval_history_path = None
        target_eval_metric_analysis = None
        best_target_wf1 = {"step": None, "weighted_f1": None, "rows": None, "pearson": None, "spearman": None}
        best_target_wf1_analysis = None
        best_target_wf1_reliability_analysis = None
        eval_base_preds = None
        eval_base_f1 = None
        eval_support = None
        eval_class_entropy = None
        eval_class_entropy_norm = None
        eval_class_mass = None
        eval_orphan = None
        eval_class_intra_distance = None
        eval_class_intra_distance_norm = None
        eval_class_reliability = None
        eval_base_counts = None
        eval_base_fracs = None
        if args.eval_target_after_train and target_has_label and target_labels is not None:
            base_eval_for_history = evaluate_head_probs(head, target_features, target_labels, report_topk=args.report_topk)
            eval_base_preds = base_eval_for_history["preds"]
            target_y_np = target_labels.detach().cpu().numpy().astype(np.int64)
            _, _, eval_base_f1, eval_support = precision_recall_fscore_support(
                target_y_np,
                eval_base_preds,
                labels=np.arange(num_classes, dtype=np.int64),
                average=None,
                zero_division=0,
            )
            eval_class_entropy, eval_class_entropy_norm, eval_class_mass, eval_orphan = class_assignment_entropy(
                A, ent, target_masses
            )
            eval_class_intra_distance = class_intra_distance
            eval_class_intra_distance_norm = class_intra_distance_norm
            eval_class_reliability = class_reliability
            eval_base_counts, eval_base_fracs = pred_distribution(eval_base_preds, num_classes)

        def record_eval_entropy_delta_f1(step, current_head):
            current_eval = evaluate_head_probs(
                current_head,
                target_features,
                target_labels if target_has_label and target_labels is not None else None,
                report_topk=args.report_topk,
            )
            im_metrics = prediction_im_metrics(current_eval["probs"])
            eval_row = {
                "step": int(step),
                "view": view,
                **im_metrics,
            }
            if target_has_label and target_labels is not None:
                eval_row.update(
                    {
                        "target_acc": current_eval.get("acc"),
                        "target_weighted_f1": current_eval.get("weighted_f1"),
                        "target_top2": current_eval.get("top2"),
                        "target_top3": current_eval.get("top3"),
                    }
                )
            target_eval_history.append(eval_row)

            if eval_base_preds is None:
                return
            current_preds = current_eval["preds"]
            current_counts, current_fracs = pred_distribution(current_preds, num_classes)
            target_y_np = target_labels.detach().cpu().numpy().astype(np.int64)
            _, _, current_f1, _ = precision_recall_fscore_support(
                target_y_np,
                current_preds,
                labels=np.arange(num_classes, dtype=np.int64),
                average=None,
                zero_division=0,
            )
            delta_f1_step = current_f1 - eval_base_f1
            valid = (eval_support > 0) & np.isfinite(eval_class_entropy) & np.isfinite(delta_f1_step)
            pearson = safe_corr(eval_class_entropy[valid], delta_f1_step[valid])
            spearman = safe_spearman(eval_class_entropy[valid], delta_f1_step[valid])
            entropy_delta_f1_correlations.append(
                {
                    "step": int(step),
                    "view": view,
                    "target_weighted_f1": current_eval.get("weighted_f1"),
                    "target_acc": current_eval.get("acc"),
                    "pearson_entropy_delta_f1_at_step": pearson,
                    "spearman_entropy_delta_f1_at_step": spearman,
                    "global_entropy": im_metrics["global_entropy"],
                    "local_entropy": im_metrics["local_entropy"],
                    "im_score": im_metrics["im_score"],
                }
            )
            step_rows = []
            for cls_idx in range(num_classes):
                row = {
                    "step": int(step),
                    "view": view,
                    "class_idx": int(cls_idx),
                    "class": idx2label[cls_idx],
                    "support": int(eval_support[cls_idx]),
                    "f1_before_step0": float(eval_base_f1[cls_idx]),
                    "f1_after": float(current_f1[cls_idx]),
                    "delta_f1_from_step0": float(delta_f1_step[cls_idx]),
                    "target_cluster_assignment_entropy": float(eval_class_entropy[cls_idx]),
                    "target_cluster_assignment_entropy_norm": float(eval_class_entropy_norm[cls_idx]),
                    "target_cluster_soft_mass": float(eval_class_mass[cls_idx]),
                    "is_orphan_class": bool(eval_orphan[cls_idx]),
                    "target_cluster_intra_distance": float(eval_class_intra_distance[cls_idx]),
                    "target_cluster_intra_distance_norm": float(eval_class_intra_distance_norm[cls_idx]),
                    "shift_reliability": float(eval_class_reliability[cls_idx]),
                    "delta_norm": float(delta_norms[cls_idx]),
                    "base_pred_count": int(eval_base_counts[cls_idx]),
                    "current_pred_count": int(current_counts[cls_idx]),
                    "base_pred_frac": float(eval_base_fracs[cls_idx]),
                    "current_pred_frac": float(current_fracs[cls_idx]),
                    "pearson_entropy_delta_f1_at_step": pearson,
                    "spearman_entropy_delta_f1_at_step": spearman,
                }
                step_rows.append(row)
                entropy_delta_f1_history.append(row)
            wf1 = current_eval.get("weighted_f1")
            if wf1 is not None and (
                best_target_wf1["weighted_f1"] is None or float(wf1) > float(best_target_wf1["weighted_f1"])
            ):
                best_target_wf1.update(
                    {
                        "step": int(step),
                        "weighted_f1": float(wf1),
                        "rows": copy.deepcopy(step_rows),
                        "pearson": pearson,
                        "spearman": spearman,
                    }
                )

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
            eval_callback=record_eval_entropy_delta_f1 if args.eval_target_after_train else None,
        )
        if entropy_delta_f1_history:
            entropy_delta_f1_history_path = os.path.join(args.output_dir, f"cct_entropy_delta_f1_history_{view}.csv")
            pd.DataFrame(entropy_delta_f1_history).to_csv(entropy_delta_f1_history_path, index=False)
        if entropy_delta_f1_correlations:
            entropy_delta_f1_correlations_path = os.path.join(
                args.output_dir, f"cct_entropy_delta_f1_correlations_{view}.csv"
            )
            pd.DataFrame(entropy_delta_f1_correlations).to_csv(entropy_delta_f1_correlations_path, index=False)
        if target_eval_history:
            target_eval_history_path, target_eval_metric_analysis = plot_target_eval_metrics(
                target_eval_history, args.output_dir, view
            )
        if best_target_wf1["rows"] is not None:
            best_df = pd.DataFrame(best_target_wf1["rows"])
            best_target_wf1_analysis = write_class_delta_plot(
                best_df,
                args.output_dir,
                view,
                "best_target_wf1",
                f"{view}: best target WF1 step {best_target_wf1['step']} entropy vs Delta F1",
            )
            best_target_wf1_reliability_analysis = write_reliability_delta_plots(
                best_df,
                args.output_dir,
                view,
                "best_target_wf1",
                f"{view}: best target WF1 step {best_target_wf1['step']} reliability diagnostics vs Delta F1",
            )

        # Target diagnostics: base vs robust (for analysis only; no target labels used in training/selection).
        base_target_metrics = None
        robust_target_metrics = None
        counts_csv_path = None
        pred_csv_path = None
        stats_path = None
        entropy_delta_f1_analysis = None
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
            pseudo_audit_cache[view] = {
                "base_preds": base_preds.copy(),
                "robust_preds": robust_preds.copy(),
                "base_probs": base_probs.copy(),
                "robust_probs": robust_probs.copy(),
            }
            if target_has_label and target_labels is not None:
                pseudo_audit_target_labels = target_labels.detach().cpu().numpy().astype(np.int64)

            if target_has_label:
                base_target_metrics = {k: base_eval.get(k) for k in ["acc", "weighted_f1", "top2", "top3"] if k in base_eval}
                robust_target_metrics = {k: robust_eval.get(k) for k in ["acc", "weighted_f1", "top2", "top3"] if k in robust_eval}
                entropy_delta_f1_analysis = write_entropy_delta_f1_diagnostics(
                    output_dir=args.output_dir,
                    view=view,
                    idx2label=idx2label,
                    target_labels=target_labels,
                    base_preds=base_preds,
                    robust_preds=robust_preds,
                    A=A,
                    cluster_entropy=ent,
                    target_masses=target_masses,
                    delta_norms=delta_norms,
                    base_counts=base_counts,
                    base_fracs=base_fracs,
                    robust_counts=robust_counts,
                    robust_fracs=robust_fracs,
                    class_intra_distance=class_intra_distance,
                    class_intra_distance_norm=class_intra_distance_norm,
                    shift_reliability=class_reliability,
                )

            pred_stats = {
                "view": view,
                "num_samples": int(target_features.shape[0]),
                "target_has_label": bool(target_has_label),
                "base_target_metrics": base_target_metrics,
                "robust_target_metrics": robust_target_metrics,
                "class_entropy_delta_f1_analysis": entropy_delta_f1_analysis,
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
        np.save(view_prefix + "_target_cluster_ids.npy", target_ids.astype(np.int64))
        np.save(view_prefix + "_target_cluster_intra_distance.npy", target_cluster_intra.astype(np.float32))
        np.save(view_prefix + "_class_target_intra_distance.npy", class_intra_distance.astype(np.float32))
        np.save(view_prefix + "_class_target_intra_distance_norm.npy", class_intra_distance_norm.astype(np.float32))
        np.save(view_prefix + "_class_shift_reliability.npy", class_reliability.astype(np.float32))
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
            **intra_distance_stats,
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
            "entropy_delta_f1_csv_path": None if entropy_delta_f1_analysis is None else entropy_delta_f1_analysis.get("csv_path"),
            "entropy_delta_f1_json_path": None if entropy_delta_f1_analysis is None else entropy_delta_f1_analysis.get("json_path"),
            "entropy_delta_f1_plot_path": None if entropy_delta_f1_analysis is None else entropy_delta_f1_analysis.get("plot_path"),
            "entropy_delta_f1_history_csv_path": entropy_delta_f1_history_path,
            "entropy_delta_f1_correlations_csv_path": entropy_delta_f1_correlations_path,
            "best_target_wf1_step": best_target_wf1["step"],
            "best_target_wf1": best_target_wf1["weighted_f1"],
            "best_target_wf1_entropy_delta_f1_csv_path": None if best_target_wf1_analysis is None else best_target_wf1_analysis.get("csv_path"),
            "best_target_wf1_entropy_delta_f1_json_path": None if best_target_wf1_analysis is None else best_target_wf1_analysis.get("json_path"),
            "best_target_wf1_entropy_delta_f1_plot_path": None if best_target_wf1_analysis is None else best_target_wf1_analysis.get("plot_path"),
            "best_target_wf1_reliability_delta_f1_csv_path": None if best_target_wf1_reliability_analysis is None else best_target_wf1_reliability_analysis.get("csv_path"),
            "best_target_wf1_reliability_delta_f1_json_path": None if best_target_wf1_reliability_analysis is None else best_target_wf1_reliability_analysis.get("json_path"),
            "best_target_wf1_reliability_delta_f1_plot_path": None if best_target_wf1_reliability_analysis is None else best_target_wf1_reliability_analysis.get("plot_path"),
            "target_eval_history_csv_path": target_eval_history_path,
            "target_eval_metric_correlations_json_path": None if target_eval_metric_analysis is None else target_eval_metric_analysis.get("json_path"),
            "target_eval_metric_plot_path": None if target_eval_metric_analysis is None else target_eval_metric_analysis.get("plot_path"),
            "pearson_im_score_target_wf1": None if target_eval_metric_analysis is None else target_eval_metric_analysis.get("pearson_im_score_target_wf1"),
            "spearman_im_score_target_wf1": None if target_eval_metric_analysis is None else target_eval_metric_analysis.get("spearman_im_score_target_wf1"),
            "pearson_entropy_delta_f1": None if entropy_delta_f1_analysis is None else entropy_delta_f1_analysis.get("pearson_entropy_delta_f1"),
            "spearman_entropy_delta_f1": None if entropy_delta_f1_analysis is None else entropy_delta_f1_analysis.get("spearman_entropy_delta_f1"),
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
            "target_cluster_intra_distance_min": cct_summary_view["target_cluster_intra_distance_min"],
            "target_cluster_intra_distance_mean": cct_summary_view["target_cluster_intra_distance_mean"],
            "target_cluster_intra_distance_max": cct_summary_view["target_cluster_intra_distance_max"],
            "class_target_intra_distance_min": cct_summary_view["class_target_intra_distance_min"],
            "class_target_intra_distance_mean": cct_summary_view["class_target_intra_distance_mean"],
            "class_target_intra_distance_max": cct_summary_view["class_target_intra_distance_max"],
            "class_shift_reliability_min": cct_summary_view["class_shift_reliability_min"],
            "class_shift_reliability_mean": cct_summary_view["class_shift_reliability_mean"],
            "class_shift_reliability_max": cct_summary_view["class_shift_reliability_max"],
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
            "cct_target_cluster_ids_path": view_prefix + "_target_cluster_ids.npy",
            "cct_target_cluster_intra_distance_path": view_prefix + "_target_cluster_intra_distance.npy",
            "cct_class_target_intra_distance_path": view_prefix + "_class_target_intra_distance.npy",
            "cct_class_target_intra_distance_norm_path": view_prefix + "_class_target_intra_distance_norm.npy",
            "cct_class_shift_reliability_path": view_prefix + "_class_shift_reliability.npy",
            "cct_source_prototypes_path": view_prefix + "_source_prototypes.npy",
            "cct_source_prototype_class_ids_path": view_prefix + "_source_prototype_class_ids.npy",
            "cct_source_prototype_masses_path": view_prefix + "_source_prototype_masses.npy",
            "target_pred_stats_path": stats_path,
            "target_pred_counts_path": counts_csv_path,
            "target_pred_csv_path": pred_csv_path,
            "entropy_delta_f1_csv_path": cct_summary_view["entropy_delta_f1_csv_path"],
            "entropy_delta_f1_json_path": cct_summary_view["entropy_delta_f1_json_path"],
            "entropy_delta_f1_plot_path": cct_summary_view["entropy_delta_f1_plot_path"],
            "entropy_delta_f1_history_csv_path": cct_summary_view["entropy_delta_f1_history_csv_path"],
            "entropy_delta_f1_correlations_csv_path": cct_summary_view["entropy_delta_f1_correlations_csv_path"],
            "best_target_wf1_step": cct_summary_view["best_target_wf1_step"],
            "best_target_wf1": cct_summary_view["best_target_wf1"],
            "best_target_wf1_entropy_delta_f1_csv_path": cct_summary_view["best_target_wf1_entropy_delta_f1_csv_path"],
            "best_target_wf1_entropy_delta_f1_json_path": cct_summary_view["best_target_wf1_entropy_delta_f1_json_path"],
            "best_target_wf1_entropy_delta_f1_plot_path": cct_summary_view["best_target_wf1_entropy_delta_f1_plot_path"],
            "best_target_wf1_reliability_delta_f1_csv_path": cct_summary_view["best_target_wf1_reliability_delta_f1_csv_path"],
            "best_target_wf1_reliability_delta_f1_json_path": cct_summary_view["best_target_wf1_reliability_delta_f1_json_path"],
            "best_target_wf1_reliability_delta_f1_plot_path": cct_summary_view["best_target_wf1_reliability_delta_f1_plot_path"],
            "target_eval_history_csv_path": cct_summary_view["target_eval_history_csv_path"],
            "target_eval_metric_correlations_json_path": cct_summary_view["target_eval_metric_correlations_json_path"],
            "target_eval_metric_plot_path": cct_summary_view["target_eval_metric_plot_path"],
            "pearson_im_score_target_wf1": cct_summary_view["pearson_im_score_target_wf1"],
            "spearman_im_score_target_wf1": cct_summary_view["spearman_im_score_target_wf1"],
            "pearson_entropy_delta_f1": cct_summary_view["pearson_entropy_delta_f1"],
            "spearman_entropy_delta_f1": cct_summary_view["spearman_entropy_delta_f1"],
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
            **intra_distance_stats,
            "pearson_entropy_delta_f1": cct_summary_view["pearson_entropy_delta_f1"],
            "spearman_entropy_delta_f1": cct_summary_view["spearman_entropy_delta_f1"],
            "entropy_delta_f1_csv_path": cct_summary_view["entropy_delta_f1_csv_path"],
            "entropy_delta_f1_plot_path": cct_summary_view["entropy_delta_f1_plot_path"],
            "entropy_delta_f1_history_csv_path": cct_summary_view["entropy_delta_f1_history_csv_path"],
            "entropy_delta_f1_correlations_csv_path": cct_summary_view["entropy_delta_f1_correlations_csv_path"],
            "best_target_wf1_step": cct_summary_view["best_target_wf1_step"],
            "best_target_wf1": cct_summary_view["best_target_wf1"],
            "best_target_wf1_entropy_delta_f1_csv_path": cct_summary_view["best_target_wf1_entropy_delta_f1_csv_path"],
            "best_target_wf1_entropy_delta_f1_plot_path": cct_summary_view["best_target_wf1_entropy_delta_f1_plot_path"],
            "best_target_wf1_reliability_delta_f1_csv_path": cct_summary_view["best_target_wf1_reliability_delta_f1_csv_path"],
            "best_target_wf1_reliability_delta_f1_plot_path": cct_summary_view["best_target_wf1_reliability_delta_f1_plot_path"],
            "target_eval_history_csv_path": cct_summary_view["target_eval_history_csv_path"],
            "target_eval_metric_plot_path": cct_summary_view["target_eval_metric_plot_path"],
            "pearson_im_score_target_wf1": cct_summary_view["pearson_im_score_target_wf1"],
            "spearman_im_score_target_wf1": cct_summary_view["spearman_im_score_target_wf1"],
        }

    pseudo_agreement_audit = write_pseudo_agreement_audit(
        args.output_dir, idx2label, pseudo_audit_cache, target_labels=pseudo_audit_target_labels
    )
    summary["pseudo_agreement_audit"] = pseudo_agreement_audit
    pseudo_gate_audit = write_pseudo_gate_audit(
        args.output_dir,
        idx2label,
        pseudo_audit_cache,
        target_labels=pseudo_audit_target_labels,
        report_topk=args.report_topk,
    )
    summary["pseudo_base_robust_gate_audit"] = pseudo_gate_audit

    out_path = os.path.join("model-classifier", f"cct_robust_classifier_summary_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[CCT] Summary saved:", out_path)


if __name__ == "__main__":
    main()
