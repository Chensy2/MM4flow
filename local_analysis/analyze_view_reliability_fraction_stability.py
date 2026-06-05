import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


MIN_PKTS = 5
MODEL_NAME = "MM4flow"
EPS = 1e-12


def parse_views(text):
    return [item.strip() for item in str(text).split(',') if item.strip()]


def parse_float_list(text):
    return [float(item) for item in str(text).split(',') if item.strip()]


def parse_int_list(text):
    return [int(item) for item in str(text).split(',') if item.strip()]


def entropy_rows(probs):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return -np.sum(probs * np.log(probs + EPS), axis=1)


def entropy_dist(dist):
    dist = np.asarray(dist, dtype=np.float64)
    dist = dist / max(float(dist.sum()), EPS)
    return float(-np.sum(dist * np.log(dist + EPS)))


def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(float(p.sum()), EPS)
    q = q / max(float(q.sum()), EPS)
    return float(np.sum(p * (np.log(p + EPS) - np.log(q + EPS))))


def js_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(float(p.sum()), EPS)
    q = q / max(float(q.sum()), EPS)
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))


def pred_frequency(pred, num_classes):
    counts = np.bincount(np.asarray(pred, dtype=int), minlength=num_classes).astype(np.float64)
    return counts / max(float(counts.sum()), EPS)


def margin_mean(probs):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape[0] == 0:
        return 0.0
    if probs.shape[1] < 2:
        return float(np.mean(probs[:, 0]))
    part = np.partition(probs, -2, axis=1)
    return float(np.mean(part[:, -1] - part[:, -2]))


def mean_confidence(probs):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape[0] == 0:
        return 0.0
    return float(np.mean(np.max(probs, axis=1)))


def mean_entropy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape[0] == 0:
        return 0.0
    return float(np.mean(entropy_rows(probs)))


def weighted_f1(y_true, pred):
    if len(y_true) == 0:
        return np.nan
    _, _, f1, _ = precision_recall_fscore_support(y_true, pred, average='weighted', zero_division=0)
    return float(f1)


def accuracy(y_true, pred):
    if len(y_true) == 0:
        return np.nan
    return float(accuracy_score(y_true, pred))


def average_ranks(values, higher_better=True):
    values = np.asarray(values, dtype=np.float64)
    finite = np.where(np.isfinite(values), values, -np.inf if higher_better else np.inf)
    order = np.argsort(-finite if higher_better else finite, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and finite[order[end]] == finite[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def rank_score_zero_one(values, higher_better=True):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return np.ones(1, dtype=np.float64)
    ranks = average_ranks(values, higher_better=higher_better)
    return (len(values) - ranks) / max(float(len(values) - 1), 1.0)


def rank_numbers(values, higher_better=True):
    ranks = average_ranks(values, higher_better=higher_better)
    return [int(rank) if abs(rank - round(rank)) < 1e-12 else float(rank) for rank in ranks]


def spearman_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return 0.0
    rx = average_ranks(x[mask], higher_better=True)
    ry = average_ranks(y[mask], higher_better=True)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def distribution_metrics(probs, pred, num_classes, source_prior, source_pred_freq=None, source_prob_mean=None):
    probs = np.asarray(probs, dtype=np.float64)
    pred = np.asarray(pred, dtype=int)
    if probs.shape[0] == 0:
        pred_freq = np.zeros(num_classes, dtype=np.float64)
        mean_prob = np.zeros(num_classes, dtype=np.float64)
    else:
        pred_freq = pred_frequency(pred, num_classes)
        mean_prob = np.mean(probs, axis=0)
    metrics = {
        'prior_penalty': kl_divergence(mean_prob, source_prior),
        'collapse_penalty': float(np.max(pred_freq)) if len(pred_freq) else 0.0,
        'prediction_diversity_risk': 1.0 - entropy_dist(pred_freq) / max(np.log(num_classes), EPS),
        'mean_confidence': mean_confidence(probs),
        'mean_margin': margin_mean(probs),
        'mean_entropy': mean_entropy(probs),
        'mean_prob': mean_prob,
        'pred_freq': pred_freq,
    }
    if source_pred_freq is not None:
        metrics['pred_freq_js_to_source'] = js_divergence(source_pred_freq, pred_freq)
    if source_prob_mean is not None:
        metrics['mean_prob_js_to_source'] = js_divergence(source_prob_mean, mean_prob)
    return metrics


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_model_label2idx(model_ts):
    info_path = os.path.join('model-classifier', model_ts, MODEL_NAME, 'info.json')
    info = load_json(info_path)
    return info['label2idx']


def label_hash_payload(label2idx):
    return json.dumps(label2idx, sort_keys=True)


def infer_source_labels(source_dataset, split_name, label2idx, expected_len):
    csv_name = 'val.csv.gz' if split_name == 'source_val' else 'train.csv.gz'
    path = os.path.join(source_dataset, csv_name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, compression='gzip', index_col=0)
    if 'label' not in df.columns:
        raise ValueError(f'Missing label column in {path}')
    raw_labels = df['label'].map(label2idx).fillna(-1).astype(int).to_numpy()
    if len(raw_labels) == expected_len:
        return raw_labels

    if 'up' in df.columns and 'down' in df.columns:
        filtered = df[df['up'] + df['down'] >= MIN_PKTS].copy()
        filtered_labels = filtered['label'].map(label2idx).fillna(-1).astype(int).to_numpy()
        if len(filtered_labels) == expected_len:
            return filtered_labels
        filtered_len = len(filtered_labels)
    else:
        filtered_len = None

    raise ValueError(
        f'Source label/cache length mismatch for {path}: '
        f'raw_labels={len(raw_labels)}, filtered_labels={filtered_len}, cache={expected_len}. '
        f'The cache may come from a different source split or preprocessing version.'
    )


def load_cache_npz(output_dir, cache_dir, split, view):
    path = os.path.join(cache_dir or os.path.join(output_dir, 'cache'), f'{split}_{view}.npz')
    if not os.path.exists(path):
        return None
    cached = np.load(path)
    return {
        'probs': cached['probs'],
        'pred': cached['pred'].astype(int),
    }


def load_output_payload(output_dir, views, source_split):
    summary_path = os.path.join(output_dir, 'summary.json')
    target_pred_path = os.path.join(output_dir, 'target_predictions.csv')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(summary_path)
    if not os.path.exists(target_pred_path):
        raise FileNotFoundError(target_pred_path)

    summary = load_json(summary_path)
    cache_dir = summary.get('cache_dir') or os.path.join(output_dir, 'cache')
    model_ts_by_view = summary['model_ts_by_view']
    label2idx = load_model_label2idx(model_ts_by_view[views[0]])
    num_classes = len(label2idx)
    source_dataset = summary['source_dataset']

    source_outputs = {}
    actual_source_split = source_split
    for view in views:
        out = load_cache_npz(output_dir, cache_dir, source_split, view)
        if out is None and source_split == 'source_val':
            out = load_cache_npz(output_dir, cache_dir, 'source', view)
            actual_source_split = 'source'
        if out is None:
            raise FileNotFoundError(
                f'Missing cache for {source_split}_{view}.npz under {cache_dir}; rerun proj_class_evidence_fusion.py.'
            )
        source_outputs[view] = out

    target_outputs = {}
    for view in views:
        out = load_cache_npz(output_dir, cache_dir, 'target', view)
        if out is None:
            raise FileNotFoundError(f'Missing cache for target_{view}.npz under {cache_dir}')
        target_outputs[view] = out

    source_labels = infer_source_labels(
        source_dataset,
        actual_source_split,
        label2idx,
        len(source_outputs[views[0]]['pred']),
    )
    pred_df = pd.read_csv(target_pred_path)
    target_labels = pred_df['y_true_idx'].fillna(-1).astype(int).to_numpy()
    target_known = target_labels >= 0
    target_labels = target_labels[target_known]
    for view in views:
        target_outputs[view]['probs'] = target_outputs[view]['probs'][target_known]
        target_outputs[view]['pred'] = target_outputs[view]['pred'][target_known]

    dataset = os.path.basename(os.path.normpath(output_dir))
    return {
        'dataset': dataset,
        'source_outputs': source_outputs,
        'target_outputs': target_outputs,
        'source_labels': source_labels,
        'target_labels': target_labels,
        'num_classes': num_classes,
    }


def stratified_indices(labels, fraction, seed):
    labels = np.asarray(labels, dtype=int)
    if fraction >= 1.0:
        return np.arange(len(labels), dtype=int)
    rng = np.random.default_rng(seed)
    selected = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        take = max(1, int(np.ceil(len(idx) * fraction)))
        take = min(take, len(idx))
        selected.append(rng.choice(idx, size=take, replace=False))
    if not selected:
        return np.asarray([], dtype=int)
    out = np.concatenate(selected)
    rng.shuffle(out)
    return out


def source_prior(labels, num_classes):
    counts = np.bincount(labels[labels >= 0], minlength=num_classes).astype(np.float64)
    return counts / max(float(counts.sum()), EPS)


def compute_source_stats(views, source_outputs, source_labels, num_classes, prior):
    stats = {}
    for view in views:
        probs = source_outputs[view]['probs']
        pred = source_outputs[view]['pred']
        metrics = distribution_metrics(probs, pred, num_classes, prior)
        stats[view] = {
            'metrics': metrics,
            'weighted_f1': weighted_f1(source_labels, pred),
            'acc': accuracy(source_labels, pred),
            'pred_freq': metrics['pred_freq'],
            'mean_prob': metrics['mean_prob'],
        }
    return stats


def compute_target_full_metrics(views, target_outputs, target_labels):
    rows = {}
    for view in views:
        pred = target_outputs[view]['pred']
        rows[view] = {
            'target_weighted_f1': weighted_f1(target_labels, pred),
        }
    return rows


def rank_aggregate(df, score_name, specs):
    components = []
    for metric, higher_better in specs:
        component = rank_score_zero_one(df[metric].to_numpy(dtype=np.float64), higher_better=higher_better)
        df[f'{score_name}_{metric}_rank_score'] = component
        components.append(component)
    df[score_name] = np.mean(np.stack(components, axis=1), axis=1)
    return df


def compute_scores_for_subset(payload, views, subset_idx, fraction, seed):
    source_outputs = payload['source_outputs']
    target_outputs = payload['target_outputs']
    source_labels = payload['source_labels']
    target_labels = payload['target_labels']
    num_classes = payload['num_classes']
    prior = source_prior(source_labels, num_classes)
    source_stats = compute_source_stats(views, source_outputs, source_labels, num_classes, prior)
    full_target_metrics = compute_target_full_metrics(views, target_outputs, target_labels)

    rows = []
    for view in views:
        src = source_stats[view]
        target_probs = target_outputs[view]['probs'][subset_idx]
        target_pred = target_outputs[view]['pred'][subset_idx]
        tgt = distribution_metrics(
            target_probs,
            target_pred,
            num_classes,
            prior,
            source_pred_freq=src['pred_freq'],
            source_prob_mean=src['mean_prob'],
        )
        target_im_score = entropy_dist(tgt['mean_prob']) - tgt['mean_entropy']
        rows.append({
            'dataset': payload['dataset'],
            'fraction': fraction,
            'seed': seed,
            'view': view,
            'target_sample_count': int(len(subset_idx)),
            'target_total_count': int(len(target_labels)),
            'target_effective_fraction': float(len(subset_idx) / max(len(target_labels), 1)),
            'target_weighted_f1': full_target_metrics[view]['target_weighted_f1'],
            'source_weighted_f1': src['weighted_f1'],
            'source_acc': src['acc'],
            'source_mean_margin': src['metrics']['mean_margin'],
            'source_mean_confidence': src['metrics']['mean_confidence'],
            'source_mean_entropy': src['metrics']['mean_entropy'],
            'delta_prior_penalty': tgt['prior_penalty'] - src['metrics']['prior_penalty'],
            'source_target_pred_freq_js': tgt['pred_freq_js_to_source'],
            'source_target_mean_prob_js': tgt['mean_prob_js_to_source'],
            'delta_collapse_penalty': tgt['collapse_penalty'] - src['metrics']['collapse_penalty'],
            'delta_prediction_diversity_risk': (
                tgt['prediction_diversity_risk'] - src['metrics']['prediction_diversity_risk']
            ),
            'entropy_shift': tgt['mean_entropy'] - src['metrics']['mean_entropy'],
            'source_target_conf_shift': tgt['mean_confidence'] - src['metrics']['mean_confidence'],
            'source_target_margin_shift': tgt['mean_margin'] - src['metrics']['mean_margin'],
            'target_mean_confidence': tgt['mean_confidence'],
            'target_mean_margin': tgt['mean_margin'],
            'target_im_score': target_im_score,
            'target_mean_entropy': tgt['mean_entropy'],
            'target_prediction_diversity_risk': tgt['prediction_diversity_risk'],
            'target_collapse_penalty': tgt['collapse_penalty'],
        })

    df = pd.DataFrame(rows)
    c_specs = [
        ('source_weighted_f1', True),
        ('source_acc', True),
        ('source_mean_margin', True),
        ('source_mean_confidence', True),
        ('source_mean_entropy', False),
    ]
    d_specs = [
        ('delta_prior_penalty', False),
        ('source_target_pred_freq_js', False),
        ('source_target_mean_prob_js', False),
        ('delta_collapse_penalty', False),
        ('delta_prediction_diversity_risk', False),
        ('entropy_shift', False),
        ('source_target_conf_shift', True),
        ('source_target_margin_shift', True),
    ]
    h_specs = [
        ('target_mean_confidence', True),
        ('target_mean_margin', True),
        ('target_im_score', True),
        ('target_mean_entropy', False),
        ('target_prediction_diversity_risk', False),
        ('target_collapse_penalty', False),
    ]
    df = rank_aggregate(df, 'C_v', c_specs)
    df = rank_aggregate(df, 'D_v', d_specs)
    df = rank_aggregate(df, 'H_v', h_specs)
    df['R_v'] = df[['C_v', 'D_v', 'H_v']].mean(axis=1)

    target_values = df['target_weighted_f1'].to_numpy(dtype=np.float64)
    best_idx = int(np.nanargmax(target_values))
    worst_idx = int(np.nanargmin(target_values))
    df['target_rank'] = rank_numbers(target_values, higher_better=True)
    for score in ['C_v', 'D_v', 'H_v', 'R_v']:
        df[f'{score[0]}_rank' if score != 'R_v' else 'R_rank'] = rank_numbers(
            df[score].to_numpy(dtype=np.float64), higher_better=True
        )
    df['is_best_view'] = False
    df.loc[best_idx, 'is_best_view'] = True
    df['is_worst_view'] = False
    df.loc[worst_idx, 'is_worst_view'] = True
    df['C_metric_count'] = len(c_specs)
    df['D_metric_count'] = len(d_specs)
    df['H_metric_count'] = len(h_specs)
    keep = [
        'dataset', 'fraction', 'seed', 'view', 'target_sample_count', 'target_total_count',
        'target_effective_fraction', 'target_weighted_f1', 'target_rank',
        'C_v', 'C_rank', 'D_v', 'D_rank', 'H_v', 'H_rank', 'R_v', 'R_rank',
        'is_best_view', 'is_worst_view', 'C_metric_count', 'D_metric_count', 'H_metric_count',
    ]
    return df[keep]


def build_summary(scores_df):
    rows = []
    for (fraction, dataset, seed), group in scores_df.groupby(['fraction', 'dataset', 'seed']):
        target = group['target_weighted_f1'].to_numpy(dtype=np.float64)
        best_idx = int(np.nanargmax(target))
        worst_idx = int(np.nanargmin(target))
        for group_name, score_col in [('C_v', 'C_v'), ('D_v', 'D_v'), ('H_v', 'H_v'), ('R_v', 'R_v')]:
            values = group[score_col].to_numpy(dtype=np.float64)
            order = np.argsort(-np.where(np.isfinite(values), values, -np.inf), kind='mergesort')
            rows.append({
                'fraction': fraction,
                'group': group_name,
                'dataset': dataset,
                'seed': seed,
                'top1_hit': bool(best_idx in set(order[:1])),
                'top2_keep': bool(best_idx in set(order[:min(2, len(order))])),
                'worst_bottom1': bool(worst_idx in set(order[-1:])),
                'worst_bottom2': bool(worst_idx in set(order[-min(2, len(order)):])),
                'spearman': spearman_corr(values, target),
            })
    per_seed = pd.DataFrame(rows)
    summary_rows = []
    for (fraction, group_name), group in per_seed.groupby(['fraction', 'group']):
        summary_rows.append({
            'fraction': fraction,
            'group': group_name,
            'dataset_seed_count': int(len(group)),
            'top1_hit_rate': float(group['top1_hit'].mean()),
            'top2_keep_rate': float(group['top2_keep'].mean()),
            'worst_bottom1_rate': float(group['worst_bottom1'].mean()),
            'worst_bottom2_rate': float(group['worst_bottom2'].mean()),
            'spearman_mean': float(group['spearman'].mean()),
            'spearman_std': float(group['spearman'].std(ddof=0)),
        })
    return pd.DataFrame(summary_rows), per_seed


def print_summary(summary_df):
    print('[View Reliability Fraction Stability]')
    print('frac     grp      top1     top2     wbot1    wbot2    sp_mean')
    for _, row in summary_df.sort_values(['fraction', 'group']).iterrows():
        print(
            f"{row['fraction']:<8.2f} {row['group']:<8} "
            f"{row['top1_hit_rate']:<8.3f} {row['top2_keep_rate']:<8.3f} "
            f"{row['worst_bottom1_rate']:<8.3f} {row['worst_bottom2_rate']:<8.3f} "
            f"{row['spearman_mean']:<8.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dirs', nargs='*', default=[])
    parser.add_argument('--glob', default=None)
    parser.add_argument('--views', default='ps,byte,mm')
    parser.add_argument('--fractions', default='0.01,0.02,0.05,0.1,0.2,0.5,1.0')
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--source_split', choices=['source_val', 'source'], default='source_val')
    parser.add_argument('--output_dir', default='local_analysis/view_reliability_fraction_outputs')
    args = parser.parse_args()

    output_dirs = list(args.output_dirs)
    if args.glob:
        output_dirs.extend(glob.glob(args.glob, recursive=True))
    output_dirs = sorted(set(output_dirs))
    if not output_dirs:
        raise ValueError('Pass --output_dirs and/or --glob.')

    os.makedirs(args.output_dir, exist_ok=True)
    views = parse_views(args.views)
    fractions = parse_float_list(args.fractions)
    seeds = parse_int_list(args.seeds)

    all_scores = []
    skipped = []
    for output_dir in output_dirs:
        try:
            payload = load_output_payload(output_dir, views, args.source_split)
            labels = payload['target_labels']
            for fraction in fractions:
                run_seeds = [0] if fraction >= 1.0 else seeds
                for seed in run_seeds:
                    subset_idx = stratified_indices(labels, fraction, seed)
                    all_scores.append(compute_scores_for_subset(payload, views, subset_idx, fraction, seed))
            print(f'[ok] {output_dir}')
        except Exception as exc:
            skipped.append({'output_dir': output_dir, 'reason': str(exc)})
            print(f'[skip] {output_dir}: {exc}')

    if not all_scores:
        raise FileNotFoundError('No valid output dirs were analyzed.')

    scores_df = pd.concat(all_scores, ignore_index=True)
    summary_df, per_seed_df = build_summary(scores_df)
    scores_df.to_csv(os.path.join(args.output_dir, 'view_reliability_fraction_scores.csv'), index=False)
    summary_df.to_csv(os.path.join(args.output_dir, 'view_reliability_fraction_summary.csv'), index=False)
    per_seed_df.to_csv(os.path.join(args.output_dir, 'view_reliability_fraction_per_seed_summary.csv'), index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(os.path.join(args.output_dir, 'view_reliability_fraction_skipped.csv'), index=False)
    print_summary(summary_df)
    print(f"[saved] {args.output_dir}")


if __name__ == '__main__':
    main()
