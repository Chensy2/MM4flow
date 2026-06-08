import argparse
import copy
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn

from proj_classifier_consensus_pseudo_finetune import (
    classifier_module,
    default_output_dir,
    device,
    ensure_dir,
    extract_features,
    load_base_model,
    make_dataset,
    model_dir_for_view,
    parse_views,
    predict_head,
    preprocess_dataframe,
    save_feature_cache,
    try_load_feature_cache,
)


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


def evaluate_head_probs(head, features, labels=None, report_topk=3):
    probs, preds = predict_head(head, features)
    out = {"probs": probs, "preds": preds}
    if labels is None:
        return out
    y_true = labels.detach().cpu().numpy().astype(np.int64)
    out["acc"] = float(accuracy_score(y_true, preds))
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="weighted", zero_division=0)
    pm, rm, f1m, _ = precision_recall_fscore_support(y_true, preds, average="macro", zero_division=0)
    out.update(
        {
            "weighted_f1": float(f1),
            "macro_f1": float(f1m),
            "weighted_precision": float(p),
            "weighted_recall": float(r),
            "macro_precision": float(pm),
            "macro_recall": float(rm),
        }
    )
    if report_topk and report_topk >= 2:
        topk = min(int(report_topk), probs.shape[1])
        top_idx = np.argsort(-probs, axis=1)[:, :topk]
        y_col = y_true.reshape(-1, 1)
        if topk >= 2:
            out["top2"] = float(np.mean(np.any(top_idx[:, :2] == y_col, axis=1)))
        if topk >= 3:
            out["top3"] = float(np.mean(np.any(top_idx[:, :3] == y_col, axis=1)))
    return out


def compute_source_mu(features, labels, num_classes):
    x = features.detach().cpu().numpy().astype(np.float32)
    y = labels.detach().cpu().numpy().astype(np.int64)
    global_mu = x.mean(axis=0).astype(np.float32)
    mu = np.zeros((num_classes, x.shape[1]), dtype=np.float32)
    counts = np.zeros((num_classes,), dtype=np.int64)
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        counts[c] = int(idx.size)
        mu[c] = x[idx].mean(axis=0).astype(np.float32) if idx.size > 0 else global_mu
    return mu, global_mu, counts


def build_consensus(preds_by_view, views):
    pred_stack = np.stack([np.asarray(preds_by_view[v], dtype=np.int64) for v in views], axis=1)
    mask = np.all(pred_stack == pred_stack[:, :1], axis=1)
    labels = pred_stack[:, 0].astype(np.int64)
    return mask, labels


def sample_budget_indices(n, ratio, strategy, seed, strata_labels=None):
    ratio = float(ratio)
    if ratio >= 1.0:
        return np.arange(n, dtype=np.int64)
    if ratio <= 0:
        raise ValueError("--target_budget_ratio must be > 0")
    m = max(1, int(round(ratio * n)))
    rng = np.random.default_rng(int(seed))
    if strategy == "random" or strata_labels is None:
        return np.sort(rng.choice(n, size=m, replace=False).astype(np.int64))
    if strategy != "stratified":
        raise ValueError(f"Unsupported target_budget_sampling={strategy}")

    labels = np.asarray(strata_labels, dtype=np.int64)
    selected = []
    classes = np.unique(labels)
    for c in classes:
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        take = int(round(m * idx.size / max(n, 1)))
        if take <= 0:
            take = 1
        take = min(take, idx.size)
        selected.extend(rng.choice(idx, size=take, replace=False).tolist())
    selected = np.asarray(sorted(set(selected)), dtype=np.int64)
    if selected.size > m:
        selected = np.sort(rng.choice(selected, size=m, replace=False).astype(np.int64))
    elif selected.size < m:
        rest = np.setdiff1d(np.arange(n, dtype=np.int64), selected, assume_unique=False)
        if rest.size > 0:
            add = rng.choice(rest, size=min(m - selected.size, rest.size), replace=False)
            selected = np.sort(np.concatenate([selected, add.astype(np.int64)]))
    return selected


def compute_global_pseudo_residual_delta(
    source_features,
    source_labels,
    target_features,
    consensus_mask,
    consensus_labels,
    budget_idx,
    num_classes,
    alpha_max,
    rho,
    min_count,
):
    source_mu, source_global_mu, source_counts = compute_source_mu(source_features, source_labels, num_classes)
    target_np = target_features.detach().cpu().numpy().astype(np.float32)
    target_budget = target_np[budget_idx]
    target_global_mu = target_budget.mean(axis=0).astype(np.float32)
    delta_global = (target_global_mu - source_global_mu).astype(np.float32)

    budget_mask = np.zeros((target_np.shape[0],), dtype=bool)
    budget_mask[budget_idx] = True
    anchor_mask = consensus_mask & budget_mask
    anchor_labels = consensus_labels[anchor_mask].astype(np.int64)
    n_k = np.bincount(anchor_labels, minlength=num_classes).astype(np.int64)
    n_anchor = int(n_k.sum())
    observed = int(np.sum(n_k > 0))
    bar_n = float(n_anchor / max(observed, 1))

    delta_pseudo = np.tile(delta_global[None, :], (num_classes, 1)).astype(np.float32)
    for c in range(num_classes):
        idx = np.where(anchor_mask & (consensus_labels == c))[0]
        if idx.size > 0:
            mu_t_c = target_np[idx].mean(axis=0).astype(np.float32)
            delta_pseudo[c] = (mu_t_c - source_mu[c]).astype(np.float32)

    alpha = np.zeros((num_classes,), dtype=np.float32)
    for c in range(num_classes):
        if n_k[c] >= int(min_count):
            alpha[c] = float(alpha_max) * float(n_k[c]) / (float(n_k[c]) + float(rho) * bar_n + 1e-12)
    delta = ((1.0 - alpha[:, None]) * delta_global[None, :] + alpha[:, None] * delta_pseudo).astype(np.float32)
    return {
        "delta": delta,
        "delta_global": delta_global,
        "delta_pseudo": delta_pseudo,
        "alpha": alpha,
        "pseudo_counts": n_k,
        "anchor_mask": anchor_mask,
        "source_mu": source_mu,
        "source_global_mu": source_global_mu,
        "target_global_mu": target_global_mu,
        "source_counts": source_counts,
        "bar_n": bar_n,
    }


def compute_global_only_delta(source_features, source_labels, target_features, budget_idx, num_classes):
    _, source_global_mu, _ = compute_source_mu(source_features, source_labels, num_classes)
    target_np = target_features.detach().cpu().numpy().astype(np.float32)
    delta_global = (target_np[budget_idx].mean(axis=0).astype(np.float32) - source_global_mu).astype(np.float32)
    return np.tile(delta_global[None, :], (num_classes, 1)).astype(np.float32)


def compute_oracle_residual_delta(source_features, source_labels, target_features, target_labels, target_known_mask, budget_idx, num_classes):
    source_mu, source_global_mu, _ = compute_source_mu(source_features, source_labels, num_classes)
    target_np = target_features.detach().cpu().numpy().astype(np.float32)
    budget_mask = np.zeros((target_np.shape[0],), dtype=bool)
    budget_mask[budget_idx] = True
    known = target_known_mask & budget_mask
    target_global_mu = target_np[budget_idx].mean(axis=0).astype(np.float32)
    delta_global = (target_global_mu - source_global_mu).astype(np.float32)
    delta = np.tile(delta_global[None, :], (num_classes, 1)).astype(np.float32)
    counts = np.zeros((num_classes,), dtype=np.int64)
    for c in range(num_classes):
        idx = np.where(known & (target_labels == c))[0]
        counts[c] = int(idx.size)
        if idx.size > 0:
            delta[c] = (target_np[idx].mean(axis=0).astype(np.float32) - source_mu[c]).astype(np.float32)
    return delta, counts


def train_shift_head(
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
    shift_rho,
):
    head = head.to(device)
    head.train()
    head_orig = copy.deepcopy({k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
    opt = torch.optim.Adam(head.parameters(), lr=lr, eps=1e-8)

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    delta_by_class = delta_by_class.to(device)

    base_val = evaluate_head(head, val_features, val_labels)
    best = {"step": 0, "metrics": base_val, "state": copy.deepcopy({k: v.detach().cpu().clone() for k, v in head.state_dict().items()}), "loss": None}
    rng = np.random.default_rng(42)
    n = int(train_features.shape[0])
    for step in range(1, int(steps) + 1):
        idx = rng.integers(low=0, high=n, size=int(batch_size), endpoint=False)
        z = train_features[idx].detach()
        y = train_labels[idx]
        z_shift = z + float(shift_rho) * delta_by_class[y]

        clean_ce = nn.functional.cross_entropy(head(z), y)
        shift_ce = nn.functional.cross_entropy(head(z_shift), y)
        anchor = torch.tensor(0.0, device=device)
        if anchor_weight > 0:
            for name, param in head.named_parameters():
                anchor = anchor + torch.sum((param - head_orig[name].to(device)) ** 2)
        loss = clean_ce + float(shift_weight) * shift_ce + float(anchor_weight) * anchor
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or (eval_every > 0 and step % int(eval_every) == 0) or step == int(steps):
            metrics = evaluate_head(head, val_features, val_labels)
            score = float(metrics.get(select_best, metrics["weighted_f1"]))
            best_score = float(best["metrics"].get(select_best, best["metrics"]["weighted_f1"]))
            if score > best_score:
                best = {
                    "step": int(step),
                    "metrics": metrics,
                    "state": copy.deepcopy({k: v.detach().cpu().clone() for k, v in head.state_dict().items()}),
                    "loss": {
                        "clean_ce": float(clean_ce.detach().cpu().item()),
                        "shift_ce": float(shift_ce.detach().cpu().item()),
                        "anchor": float(anchor.detach().cpu().item()),
                        "total": float(loss.detach().cpu().item()),
                    },
                }
    head.load_state_dict(best["state"])
    return base_val, best


def class_diagnostic_rows(idx2label, pseudo_counts, alpha, target_labels, target_known_mask, consensus_mask, consensus_labels):
    rows = []
    num_classes = len(idx2label)
    has_labels = target_labels is not None and target_known_mask is not None and bool(target_known_mask.any())
    for c in range(num_classes):
        row = {
            "class_idx": int(c),
            "class": idx2label[c],
            "pseudo_count": int(pseudo_counts[c]),
            "alpha_k": float(alpha[c]),
            "target_support_diagnostic_only": None,
            "pseudo_accuracy_diagnostic_only": None,
        }
        if has_labels:
            row["target_support_diagnostic_only"] = int(np.sum(target_known_mask & (target_labels == c)))
            m = consensus_mask & target_known_mask & (consensus_labels == c)
            row["pseudo_accuracy_diagnostic_only"] = None if int(m.sum()) == 0 else float(np.mean(target_labels[m] == c))
        rows.append(row)
    return rows


def stats_min_mean_max(x, prefix):
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return {f"{prefix}_min": 0.0, f"{prefix}_mean": 0.0, f"{prefix}_max": 0.0}
    return {f"{prefix}_min": float(np.min(arr)), f"{prefix}_mean": float(np.mean(arr)), f"{prefix}_max": float(np.max(arr))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", required=True)
    parser.add_argument("--target_csv", required=True)
    parser.add_argument("--views", default="ps,byte,mm")
    parser.add_argument("--ps_model_ts", default=None)
    parser.add_argument("--byte_model_ts", default=None)
    parser.add_argument("--mm_model_ts", default=None)
    parser.add_argument("--output_suffix", default="_global_pseudo_residual_cls")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--adapt_strategy", choices=["global_pseudo_residual"], default="global_pseudo_residual")

    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--select_best", choices=["weighted_f1", "macro_f1", "acc"], default="weighted_f1")
    parser.add_argument("--anchor_weight", type=float, default=1e-3)

    parser.add_argument("--pseudo_residual_alpha_max", type=float, default=0.5)
    parser.add_argument("--pseudo_residual_rho", type=float, default=1.0)
    parser.add_argument("--pseudo_residual_min_count", type=int, default=3)
    parser.add_argument("--pseudo_residual_shift_rho", type=float, default=0.5)
    parser.add_argument("--pseudo_residual_shift_weight", type=float, default=1.0)
    parser.add_argument("--target_budget_ratio", type=float, default=1.0)
    parser.add_argument("--target_budget_sampling", choices=["stratified", "random"], default="stratified")
    parser.add_argument("--target_budget_seed", type=int, default=128)

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
        args.output_dir = default_output_dir(args.source_dataset, args.target_csv, views).replace("consensus_", "global_pseudo_residual_")
    ensure_dir(args.output_dir)

    base_view = "ps" if "ps" in views else views[0]
    _, _, _, base_info = load_base_model(base_view, model_ts_by_view[base_view])
    label2idx = base_info["label2idx"]
    idx2label = {int(v): k for k, v in label2idx.items()}
    idx2label = dict(sorted(idx2label.items(), key=lambda item: item[0]))
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
    if args.max_target_features is not None and len(df_target_all) > args.max_target_features:
        df_target_all = df_target_all.head(args.max_target_features).copy()

    target_known_mask = None
    target_labels_np = None
    target_labels_torch = None
    if "label" in df_target_all.columns:
        mapped = df_target_all["label"].map(label2idx)
        target_known_mask = mapped.notna().to_numpy()
        target_labels_np = mapped.fillna(-1).astype(np.int64).to_numpy()
        target_labels_torch = torch.from_numpy(target_labels_np.astype(np.int64))

    models, infos = {}, {}
    source_features, source_labels = {}, {}
    val_features, val_labels = {}, {}
    target_features, target_probs, target_preds = {}, {}, {}

    for view in views:
        print(f"[GlobalPseudoResidual] Loading/extracting view={view}")
        model, _, _, info = load_base_model(view, model_ts_by_view[view])
        if info["label2idx"] != label2idx:
            raise ValueError(f"{view} label2idx does not match {base_view}.")
        models[view] = model
        infos[view] = info

        df_train = preprocess_dataframe(df_train_all, view, label2idx=label2idx)
        df_val = preprocess_dataframe(df_val_all, view, label2idx=label2idx)
        df_target = preprocess_dataframe(df_target_all, view, label2idx=None)

        reuse_train = try_load_feature_cache(args.reuse_feature_cache_dir, "source_train", view, require_labels=True)
        if reuse_train is None:
            train_ds = make_dataset(df_train, view, has_label=True)
            feat, lab = extract_features(model, train_ds, args.infer_batch_size, has_label=True)
            save_feature_cache(args.save_feature_cache_dir, "source_train", view, feat.numpy(), labels=lab.numpy())
        else:
            feat, lab = reuse_train["features"], reuse_train["labels"]
        reuse_val = try_load_feature_cache(args.reuse_feature_cache_dir, "source_val", view, require_labels=True)
        if reuse_val is None:
            val_ds = make_dataset(df_val, view, has_label=True)
            vfeat, vlab = extract_features(model, val_ds, args.infer_batch_size, has_label=True)
            save_feature_cache(args.save_feature_cache_dir, "source_val", view, vfeat.numpy(), labels=vlab.numpy())
        else:
            vfeat, vlab = reuse_val["features"], reuse_val["labels"]

        if args.max_train_features is not None and feat.shape[0] > args.max_train_features:
            feat, lab = feat[: args.max_train_features], lab[: args.max_train_features]
        if args.max_val_features is not None and vfeat.shape[0] > args.max_val_features:
            vfeat, vlab = vfeat[: args.max_val_features], vlab[: args.max_val_features]

        reuse_target = try_load_feature_cache(args.reuse_feature_cache_dir, "target", view, require_pred_prob=False)
        if reuse_target is None:
            target_ds = make_dataset(df_target, view, has_label=False)
            tfeat, _ = extract_features(model, target_ds, args.infer_batch_size, has_label=False)
        else:
            tfeat = reuse_target["features"]
        head = classifier_module(model, view)
        prob, pred = predict_head(head, tfeat)
        save_feature_cache(args.save_feature_cache_dir, "target", view, tfeat.numpy(), pred=pred, prob=prob)

        source_features[view], source_labels[view] = feat, lab
        val_features[view], val_labels[view] = vfeat, vlab
        target_features[view], target_probs[view], target_preds[view] = tfeat, prob, pred

    target_len = min(len(target_preds[v]) for v in views)
    for view in views:
        target_features[view] = target_features[view][:target_len]
        target_probs[view] = target_probs[view][:target_len]
        target_preds[view] = target_preds[view][:target_len]
    if target_known_mask is not None:
        target_known_mask = target_known_mask[:target_len]
        target_labels_np = target_labels_np[:target_len]
        target_labels_torch = torch.from_numpy(target_labels_np.astype(np.int64))

    consensus_mask, consensus_labels = build_consensus(target_preds, views)
    budget_idx = sample_budget_indices(
        target_len,
        args.target_budget_ratio,
        args.target_budget_sampling,
        args.target_budget_seed,
        strata_labels=consensus_labels,
    )
    budget_mask = np.zeros((target_len,), dtype=bool)
    budget_mask[budget_idx] = True
    consensus_budget_mask = consensus_mask & budget_mask

    summary = {
        "method": "global_pseudo_residual_shift",
        "adapt_strategy": args.adapt_strategy,
        "source_dataset": args.source_dataset,
        "target_csv": args.target_csv,
        "views": views,
        "device": device,
        "steps": int(args.steps),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "infer_batch_size": int(args.infer_batch_size),
        "eval_every": int(args.eval_every),
        "select_best": args.select_best,
        "anchor_weight": float(args.anchor_weight),
        "pseudo_residual_alpha_max": float(args.pseudo_residual_alpha_max),
        "pseudo_residual_rho": float(args.pseudo_residual_rho),
        "pseudo_residual_min_count": int(args.pseudo_residual_min_count),
        "pseudo_residual_shift_rho": float(args.pseudo_residual_shift_rho),
        "pseudo_residual_shift_weight": float(args.pseudo_residual_shift_weight),
        "target_budget_ratio": float(args.target_budget_ratio),
        "target_budget_sampling": args.target_budget_sampling,
        "target_budget_seed": int(args.target_budget_seed),
        "num_target_samples": int(target_len),
        "target_budget_count": int(budget_idx.size),
        "pseudo_anchor_coverage": float(np.sum(consensus_budget_mask) / max(target_len, 1)),
        "output_suffix": args.output_suffix,
        "output_dir": args.output_dir,
        "per_view": {},
    }

    for view in views:
        print(f"[GlobalPseudoResidual] Computing residual delta and training view={view}")
        residual = compute_global_pseudo_residual_delta(
            source_features[view],
            source_labels[view],
            target_features[view],
            consensus_mask,
            consensus_labels,
            budget_idx,
            num_classes,
            args.pseudo_residual_alpha_max,
            args.pseudo_residual_rho,
            args.pseudo_residual_min_count,
        )
        delta = residual["delta"]
        alpha = residual["alpha"]
        pseudo_counts = residual["pseudo_counts"]
        delta_global = residual["delta_global"]
        delta_global_only = compute_global_only_delta(
            source_features[view], source_labels[view], target_features[view], budget_idx, num_classes
        )
        oracle_delta, oracle_counts = (None, None)
        if target_known_mask is not None and target_labels_np is not None and bool(target_known_mask.any()):
            oracle_delta, oracle_counts = compute_oracle_residual_delta(
                source_features[view],
                source_labels[view],
                target_features[view],
                target_labels_np,
                target_known_mask,
                budget_idx,
                num_classes,
            )

        for p in models[view].parameters():
            p.requires_grad_(False)
        head = classifier_module(models[view], view)
        for p in head.parameters():
            p.requires_grad_(True)
        base_head_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
        base_val, best = train_shift_head(
            head,
            source_features[view],
            source_labels[view],
            val_features[view],
            val_labels[view],
            torch.from_numpy(delta),
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            select_best=args.select_best,
            shift_weight=args.pseudo_residual_shift_weight,
            anchor_weight=args.anchor_weight,
            shift_rho=args.pseudo_residual_shift_rho,
        )

        base_target_metrics = None
        robust_target_metrics = None
        global_only_target_metrics = None
        oracle_target_metrics = None
        if args.eval_target_after_train and target_labels_torch is not None and bool(target_known_mask.any()):
            eval_features = target_features[view][target_known_mask]
            eval_labels = torch.from_numpy(target_labels_np[target_known_mask].astype(np.int64))
            base_head = copy.deepcopy(head).to(device)
            base_head.load_state_dict(base_head_state, strict=True)
            base_target_metrics = {
                k: v for k, v in evaluate_head_probs(base_head, eval_features, eval_labels, args.report_topk).items() if k != "probs" and k != "preds"
            }
            robust_target_metrics = {
                k: v for k, v in evaluate_head_probs(head, eval_features, eval_labels, args.report_topk).items() if k != "probs" and k != "preds"
            }
            # Diagnostic-only upper/lower comparisons: train throwaway heads from the original state.
            global_head = copy.deepcopy(head).to(device)
            global_head.load_state_dict(base_head_state, strict=True)
            train_shift_head(
                global_head,
                source_features[view],
                source_labels[view],
                val_features[view],
                val_labels[view],
                torch.from_numpy(delta_global_only),
                steps=args.steps,
                lr=args.lr,
                batch_size=args.batch_size,
                eval_every=args.eval_every,
                select_best=args.select_best,
                shift_weight=args.pseudo_residual_shift_weight,
                anchor_weight=args.anchor_weight,
                shift_rho=args.pseudo_residual_shift_rho,
            )
            global_only_target_metrics = {
                k: v for k, v in evaluate_head_probs(global_head, eval_features, eval_labels, args.report_topk).items() if k != "probs" and k != "preds"
            }
            if oracle_delta is not None:
                oracle_head = copy.deepcopy(head).to(device)
                oracle_head.load_state_dict(base_head_state, strict=True)
                train_shift_head(
                    oracle_head,
                    source_features[view],
                    source_labels[view],
                    val_features[view],
                    val_labels[view],
                    torch.from_numpy(oracle_delta),
                    steps=args.steps,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    eval_every=args.eval_every,
                    select_best=args.select_best,
                    shift_weight=args.pseudo_residual_shift_weight,
                    anchor_weight=args.anchor_weight,
                    shift_rho=args.pseudo_residual_shift_rho,
                )
                oracle_target_metrics = {
                    k: v for k, v in evaluate_head_probs(oracle_head, eval_features, eval_labels, args.report_topk).items() if k != "probs" and k != "preds"
                }

        global_pseudo_residual_anchor_count = int(np.sum(pseudo_counts))
        observed = int(np.sum(pseudo_counts > 0))
        residual_count = int(np.sum(alpha > 0))
        zero_support = int(np.sum(pseudo_counts == 0))
        global_only = int(np.sum(alpha <= 0))
        pseudo_acc_diag = None
        if target_known_mask is not None and target_labels_np is not None:
            m = consensus_budget_mask & target_known_mask
            pseudo_acc_diag = None if int(m.sum()) == 0 else float(np.mean(consensus_labels[m] == target_labels_np[m]))

        class_rows = class_diagnostic_rows(
            idx2label, pseudo_counts, alpha, target_labels_np, target_known_mask, consensus_budget_mask, consensus_labels
        )
        class_csv = os.path.join(args.output_dir, f"global_pseudo_residual_class_diagnostics_{view}.csv")
        pd.DataFrame(class_rows).to_csv(class_csv, index=False)
        prefix = os.path.join(args.output_dir, f"global_pseudo_residual_{view}")
        np.save(prefix + "_delta.npy", delta.astype(np.float32))
        np.save(prefix + "_delta_global.npy", delta_global.astype(np.float32))
        np.save(prefix + "_delta_pseudo.npy", residual["delta_pseudo"].astype(np.float32))
        np.save(prefix + "_alpha.npy", alpha.astype(np.float32))
        np.save(prefix + "_pseudo_counts.npy", pseudo_counts.astype(np.int64))
        if oracle_delta is not None:
            np.save(prefix + "_oracle_delta.npy", oracle_delta.astype(np.float32))
            np.save(prefix + "_oracle_counts.npy", oracle_counts.astype(np.int64))

        delta_norm = np.linalg.norm(delta, axis=1)
        view_summary = {
            "base_model_ts": model_ts_by_view[view],
            "global_pseudo_residual_anchor_count": global_pseudo_residual_anchor_count,
            "global_pseudo_residual_observed_class_count": observed,
            "global_pseudo_residual_zero_support_class_count": zero_support,
            "global_pseudo_residual_residual_class_count": residual_count,
            "global_pseudo_residual_global_only_class_count": global_only,
            "global_pseudo_residual_mean_anchor_count": float(residual["bar_n"]),
            "alpha_min": float(np.min(alpha)) if alpha.size else 0.0,
            "alpha_mean": float(np.mean(alpha)) if alpha.size else 0.0,
            "alpha_max": float(np.max(alpha)) if alpha.size else 0.0,
            "class_counts": {idx2label[i]: int(pseudo_counts[i]) for i in range(num_classes)},
            "alpha_k": {idx2label[i]: float(alpha[i]) for i in range(num_classes)},
            "global_delta_norm": float(np.linalg.norm(delta_global)),
            **stats_min_mean_max(delta_norm, "class_delta_norm"),
            "pseudo_anchor_coverage": float(global_pseudo_residual_anchor_count / max(target_len, 1)),
            "pseudo_anchor_accuracy_diagnostic": pseudo_acc_diag,
            "class_diagnostics_csv": class_csv,
            "delta_path": prefix + "_delta.npy",
            "delta_global_path": prefix + "_delta_global.npy",
            "delta_pseudo_path": prefix + "_delta_pseudo.npy",
            "alpha_path": prefix + "_alpha.npy",
            "pseudo_counts_path": prefix + "_pseudo_counts.npy",
            "base_val_metrics": base_val,
            "best_step": int(best["step"]),
            "best_val_metrics": best["metrics"],
            "best_loss": best["loss"],
            "base_target_metrics_diagnostic": base_target_metrics,
            "global_shift_only_target_metrics_diagnostic": global_only_target_metrics,
            "global_pseudo_residual_target_metrics_diagnostic": robust_target_metrics,
            "oracle_class_residual_target_metrics_diagnostic": oracle_target_metrics,
        }
        with open(os.path.join(args.output_dir, f"global_pseudo_residual_summary_{view}.json"), "w") as f:
            json.dump(view_summary, f, indent=2, sort_keys=True)

        output, run_id = model_ts_by_view[view].split("/", 1)
        new_model_ts = f"{output}/{run_id}{args.output_suffix}"
        new_root = os.path.join("model-classifier", new_model_ts, "MM4flow")
        ensure_dir(new_root)
        new_model_dir = model_dir_for_view(new_root, view)
        ensure_dir(new_model_dir)
        info = dict(infos[view])
        info["timestamp"] = datetime.now().strftime("%Y%m%d%H%M")
        info["global_pseudo_residual_classifier"] = dict(view_summary)
        with open(os.path.join(new_root, "info.json"), "w") as f:
            json.dump(info, f, indent=2, sort_keys=True)
        models[view] = models[view].to("cpu")
        torch.save(models[view].state_dict(), os.path.join(new_model_dir, "pytorch_model.bin"))

        view_summary["output_model_ts"] = new_model_ts
        summary["per_view"][view] = view_summary

    summary_path = os.path.join(args.output_dir, "global_pseudo_residual_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    global_summary = os.path.join(
        "model-classifier", f"global_pseudo_residual_summary_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    )
    with open(global_summary, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[GlobalPseudoResidual] Summary saved:", summary_path)
    print("[GlobalPseudoResidual] Global summary saved:", global_summary)


if __name__ == "__main__":
    main()
