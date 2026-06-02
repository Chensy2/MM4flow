import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


DEFAULT_LAMBDAS = [0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_TAUS = [0.25, 0.5, 1.0]


def parse_float_list(text):
    return [float(item) for item in str(text).split(',') if str(item).strip()]


def softmax_np(values, tau=1.0):
    values = np.asarray(values, dtype=np.float64)
    z = values / max(float(tau), 1e-12)
    z = z - np.max(z)
    exp = np.exp(z)
    return exp / max(float(exp.sum()), 1e-12)


def infer_setting_name(output_dir):
    return os.path.basename(os.path.normpath(output_dir))


def load_views(output_dir, pred_df, reliability_df):
    if reliability_df is not None and 'view' in reliability_df:
        return reliability_df['view'].astype(str).tolist()
    views = []
    excluded = {
        'avg', 'prob_only', 'geometry', 'agreement_gated', 'hspf',
        'class_oracle', 'sample_oracle', 'robust_view_first',
    }
    for col in pred_df.columns:
        if col.startswith('pred_') and col.endswith('_idx'):
            name = col[len('pred_'):-len('_idx')]
            if name not in excluded:
                views.append(name)
    if not views:
        raise ValueError(f'Cannot infer views from {output_dir}/target_predictions.csv')
    return views


def prob_columns_for_view(pred_df, view):
    prefix = f'prob_{view}_'
    cols = [col for col in pred_df.columns if col.startswith(prefix)]
    if not cols:
        raise ValueError(f'Missing probability columns for view={view}')
    return cols


def load_probs(pred_df, views):
    cols_by_view = {view: prob_columns_for_view(pred_df, view) for view in views}
    first_cols = cols_by_view[views[0]]
    label_suffixes = [col[len(f'prob_{views[0]}_'):] for col in first_cols]
    probs = {}
    for view in views:
        cols = [f'prob_{view}_{label}' for label in label_suffixes]
        missing = [col for col in cols if col not in pred_df.columns]
        if missing:
            raise ValueError(f'Missing probability columns for {view}: {missing[:5]}')
        probs[view] = pred_df[cols].to_numpy(dtype=np.float64)
    return probs, label_suffixes


def known_labels(pred_df):
    y = pred_df['y_true_idx'].fillna(-1).astype(int).to_numpy()
    mask = y >= 0
    return y, mask


def metrics_from_pred(y_true, pred, known_mask):
    if known_mask.sum() == 0:
        return None, None, None
    y = y_true[known_mask]
    p = pred[known_mask]
    acc = float(accuracy_score(y, p))
    _, _, weighted_f1, _ = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    return acc, float(weighted_f1), float(macro_f1)


def metrics_from_probs(y_true, probs, known_mask):
    return metrics_from_pred(y_true, np.asarray(probs).argmax(axis=1), known_mask)


def normalize_probs(probs):
    probs = np.asarray(probs, dtype=np.float64)
    return probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)


def average_probs(probs_by_view, views):
    return normalize_probs(np.mean([probs_by_view[view] for view in views], axis=0))


def weighted_probs(probs_by_view, views, weights):
    fused = np.zeros_like(probs_by_view[views[0]], dtype=np.float64)
    for view, weight in zip(views, weights):
        fused += float(weight) * probs_by_view[view]
    return normalize_probs(fused)


def add_result(rows, dataset, method, probs, y_true, known_mask, anchor_view, lam, tau,
               d_top, d_second, d_gap, anchor_is_best, view_weights,
               avg_f1, best_single_f1):
    acc, weighted_f1, macro_f1 = metrics_from_probs(y_true, probs, known_mask)
    rows.append({
        'dataset': dataset,
        'method': method,
        'anchor_view': anchor_view,
        'lambda': lam,
        'tau': tau,
        'D_top': d_top,
        'D_second': d_second,
        'D_gap': d_gap,
        'anchor_is_best': bool(anchor_is_best),
        'acc': acc,
        'weighted_f1': weighted_f1,
        'macro_f1': macro_f1,
        'gain_vs_avg': None if weighted_f1 is None or avg_f1 is None else weighted_f1 - avg_f1,
        'regret_to_best_single': None if weighted_f1 is None or best_single_f1 is None else best_single_f1 - weighted_f1,
        'view_weights': json.dumps(view_weights, sort_keys=True),
    })


def add_pred_result(rows, dataset, method, pred, y_true, known_mask, anchor_view, d_top,
                    d_second, d_gap, anchor_is_best, avg_f1, best_single_f1):
    acc, weighted_f1, macro_f1 = metrics_from_pred(y_true, pred, known_mask)
    rows.append({
        'dataset': dataset,
        'method': method,
        'anchor_view': anchor_view,
        'lambda': None,
        'tau': None,
        'D_top': d_top,
        'D_second': d_second,
        'D_gap': d_gap,
        'anchor_is_best': bool(anchor_is_best),
        'acc': acc,
        'weighted_f1': weighted_f1,
        'macro_f1': macro_f1,
        'gain_vs_avg': None if weighted_f1 is None or avg_f1 is None else weighted_f1 - avg_f1,
        'regret_to_best_single': None if weighted_f1 is None or best_single_f1 is None else best_single_f1 - weighted_f1,
        'view_weights': '{}',
    })


def evaluate_output_dir(output_dir, lambdas, taus):
    pred_path = os.path.join(output_dir, 'target_predictions.csv')
    rel_path = os.path.join(output_dir, 'view_reliability_diagnostics.csv')
    if not os.path.exists(pred_path):
        raise FileNotFoundError(pred_path)
    if not os.path.exists(rel_path):
        raise FileNotFoundError(f'Missing {rel_path}; rerun proj_class_evidence_fusion.py first.')

    dataset = infer_setting_name(output_dir)
    pred_df = pd.read_csv(pred_path)
    reliability_df = pd.read_csv(rel_path)
    views = load_views(output_dir, pred_df, reliability_df)
    probs_by_view, _ = load_probs(pred_df, views)
    y_true, known_mask = known_labels(pred_df)

    d_by_view = dict(zip(reliability_df['view'].astype(str), reliability_df['D_v'].astype(float)))
    missing_d = [view for view in views if view not in d_by_view]
    if missing_d:
        raise ValueError(f'Missing D_v for views: {missing_d}')

    view_metrics = {}
    for view in views:
        acc, weighted_f1, macro_f1 = metrics_from_probs(y_true, probs_by_view[view], known_mask)
        view_metrics[view] = {'acc': acc, 'weighted_f1': weighted_f1, 'macro_f1': macro_f1}

    best_view = max(views, key=lambda view: -np.inf if view_metrics[view]['weighted_f1'] is None else view_metrics[view]['weighted_f1'])
    best_single_f1 = view_metrics[best_view]['weighted_f1']
    d_sorted = sorted(views, key=lambda view: d_by_view[view], reverse=True)
    anchor = d_sorted[0]
    d_top = float(d_by_view[anchor])
    d_second = float(d_by_view[d_sorted[1]]) if len(d_sorted) > 1 else 0.0
    d_gap = d_top - d_second
    anchor_is_best = anchor == best_view

    rows = []
    avg_probs = average_probs(probs_by_view, views)
    _, avg_f1, _ = metrics_from_probs(y_true, avg_probs, known_mask)

    add_result(
        rows, dataset, 'average_softmax_all_views', avg_probs, y_true, known_mask,
        anchor, None, None, d_top, d_second, d_gap, anchor_is_best,
        {view: 1.0 / len(views) for view in views}, avg_f1, best_single_f1
    )
    add_result(
        rows, dataset, 'best_single_oracle', probs_by_view[best_view], y_true, known_mask,
        anchor, None, None, d_top, d_second, d_gap, anchor_is_best,
        {best_view: 1.0}, avg_f1, best_single_f1
    )
    add_result(
        rows, dataset, 'D_anchor_only', probs_by_view[anchor], y_true, known_mask,
        anchor, None, None, d_top, d_second, d_gap, anchor_is_best,
        {anchor: 1.0}, avg_f1, best_single_f1
    )

    for tau in taus:
        weights = softmax_np([d_by_view[view] for view in views], tau=tau)
        add_result(
            rows, dataset, 'softmax_D_weighted_all_views', weighted_probs(probs_by_view, views, weights),
            y_true, known_mask, anchor, None, tau, d_top, d_second, d_gap, anchor_is_best,
            {view: float(weight) for view, weight in zip(views, weights)}, avg_f1, best_single_f1
        )

    residual_views = [view for view in views if view != anchor]
    for lam in lambdas:
        if residual_views:
            residual = average_probs(probs_by_view, residual_views)
            probs = normalize_probs(lam * probs_by_view[anchor] + (1.0 - lam) * residual)
            weights = {anchor: float(lam)}
            for view in residual_views:
                weights[view] = float((1.0 - lam) / len(residual_views))
        else:
            probs = probs_by_view[anchor]
            weights = {anchor: 1.0}
        add_result(
            rows, dataset, 'anchor_uniform_residual', probs, y_true, known_mask,
            anchor, lam, None, d_top, d_second, d_gap, anchor_is_best,
            weights, avg_f1, best_single_f1
        )

        for tau in taus:
            if residual_views:
                residual_weights = softmax_np([d_by_view[view] for view in residual_views], tau=tau)
                residual = weighted_probs(probs_by_view, residual_views, residual_weights)
                probs = normalize_probs(lam * probs_by_view[anchor] + (1.0 - lam) * residual)
                weights = {anchor: float(lam)}
                for view, weight in zip(residual_views, residual_weights):
                    weights[view] = float((1.0 - lam) * weight)
            else:
                probs = probs_by_view[anchor]
                weights = {anchor: 1.0}
            add_result(
                rows, dataset, 'anchor_D_weighted_residual', probs, y_true, known_mask,
                anchor, lam, tau, d_top, d_second, d_gap, anchor_is_best,
                weights, avg_f1, best_single_f1
            )

    existing_methods = {
        'pred_prob_only_idx': 'prob_class',
        'pred_geometry_idx': 'geometry_class',
        'pred_agreement_gated_idx': 'agreement_gate',
        'pred_hspf_idx': 'HSPF',
        'pred_class_oracle_idx': 'class_oracle',
        'pred_robust_view_first_idx': 'robust_view_first',
    }
    for col, method in existing_methods.items():
        if col in pred_df.columns:
            add_pred_result(
                rows, dataset, method, pred_df[col].astype(int).to_numpy(),
                y_true, known_mask, anchor, d_top, d_second, d_gap,
                anchor_is_best, avg_f1, best_single_f1
            )

    results = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, 'degradation_anchor_residual_results.csv')
    results.to_csv(out_path, index=False)

    diagnostics = build_dataset_diagnostics(results)
    diagnostics.to_csv(os.path.join(output_dir, 'degradation_anchor_residual_diagnostics.csv'), index=False)
    return results


def best_method_value(df, method):
    subset = df[df['method'] == method]
    if subset.empty:
        return None
    return float(subset['weighted_f1'].max())


def build_dataset_diagnostics(results):
    rows = []
    for dataset, df in results.groupby('dataset'):
        avg = best_method_value(df, 'average_softmax_all_views')
        anchor = best_method_value(df, 'D_anchor_only')
        uniform = best_method_value(df, 'anchor_uniform_residual')
        dres = best_method_value(df, 'anchor_D_weighted_residual')
        best_single = best_method_value(df, 'best_single_oracle')
        anchor_is_best = bool(df['anchor_is_best'].iloc[0])
        rows.append({
            'dataset': dataset,
            'anchor_is_best': anchor_is_best,
            'D_anchor_only_beats_average': None if anchor is None or avg is None else anchor > avg,
            'best_uniform_residual_beats_D_anchor_only': None if uniform is None or anchor is None else uniform > anchor,
            'best_D_weighted_residual_beats_best_uniform_residual': None if dres is None or uniform is None else dres > uniform,
            'residual_lowers_regret_when_anchor_wrong': None if anchor_is_best or best_single is None or anchor is None or max(uniform or -np.inf, dres or -np.inf) == -np.inf else (
                best_single - max(uniform or -np.inf, dres or -np.inf) < best_single - anchor
            ),
            'residual_pollutes_anchor_when_anchor_correct': None if not anchor_is_best or anchor is None or max(uniform or -np.inf, dres or -np.inf) == -np.inf else (
                max(uniform or -np.inf, dres or -np.inf) < anchor
            ),
            'average_weighted_f1': avg,
            'D_anchor_only_weighted_f1': anchor,
            'best_uniform_residual_weighted_f1': uniform,
            'best_D_weighted_residual_weighted_f1': dres,
            'best_single_oracle_weighted_f1': best_single,
        })
    return pd.DataFrame(rows)


def aggregate_results(all_results, output_csv):
    if all_results.empty:
        return
    baseline_avg = all_results[all_results['method'] == 'average_softmax_all_views'][['dataset', 'weighted_f1']]
    baseline_avg = baseline_avg.rename(columns={'weighted_f1': 'avg_f1'})
    baseline_anchor = all_results[all_results['method'] == 'D_anchor_only'][['dataset', 'weighted_f1']]
    baseline_anchor = baseline_anchor.rename(columns={'weighted_f1': 'anchor_f1'})
    baseline_dall = all_results[all_results['method'] == 'softmax_D_weighted_all_views']
    baseline_dall = baseline_dall.groupby('dataset', as_index=False)['weighted_f1'].max()
    baseline_dall = baseline_dall.rename(columns={'weighted_f1': 'd_all_f1'})

    merged = all_results.merge(baseline_avg, on='dataset', how='left')
    merged = merged.merge(baseline_anchor, on='dataset', how='left')
    merged = merged.merge(baseline_dall, on='dataset', how='left')
    merged['method_key'] = merged.apply(
        lambda row: f"{row['method']}_lambda={row['lambda']}_tau={row['tau']}",
        axis=1,
    )

    rows = []
    group_cols = ['method', 'lambda', 'tau']
    for keys, group in merged.groupby(group_cols, dropna=False):
        method, lam, tau = keys
        rows.append({
            'method': method,
            'lambda': lam,
            'tau': tau,
            'num_datasets': int(len(group)),
            'mean_weighted_f1': float(group['weighted_f1'].mean()),
            'mean_gain_vs_average': float((group['weighted_f1'] - group['avg_f1']).mean()),
            'mean_regret_to_best_single': float(group['regret_to_best_single'].mean()),
            'win_rate_vs_average_softmax': float((group['weighted_f1'] > group['avg_f1']).mean()),
            'win_rate_vs_D_anchor_only': float((group['weighted_f1'] > group['anchor_f1']).mean()),
            'win_rate_vs_softmax_D_weighted_all_views': float((group['weighted_f1'] > group['d_all_f1']).mean()),
            'mean_performance_anchor_is_best_true': float(group[group['anchor_is_best']]['weighted_f1'].mean()) if group['anchor_is_best'].any() else None,
            'mean_performance_anchor_is_best_false': float(group[~group['anchor_is_best']]['weighted_f1'].mean()) if (~group['anchor_is_best']).any() else None,
        })
    aggregate = pd.DataFrame(rows).sort_values(['mean_weighted_f1', 'method'], ascending=[False, True])
    aggregate.to_csv(output_csv, index=False)
    all_results.to_csv(output_csv.replace('.csv', '_all_results.csv'), index=False)
    build_dataset_diagnostics(all_results).to_csv(output_csv.replace('.csv', '_dataset_diagnostics.csv'), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dirs', nargs='*', default=[])
    parser.add_argument('--glob', default=None)
    parser.add_argument('--lambdas', default=','.join(str(x) for x in DEFAULT_LAMBDAS))
    parser.add_argument('--taus', default=','.join(str(x) for x in DEFAULT_TAUS))
    parser.add_argument('--aggregate_csv', default='degradation_anchor_residual_aggregate_summary.csv')
    args = parser.parse_args()

    output_dirs = list(args.output_dirs)
    if args.glob:
        output_dirs.extend(glob.glob(args.glob, recursive=True))
    output_dirs = sorted(set(output_dirs))
    if not output_dirs:
        raise ValueError('Pass --output_dirs and/or --glob.')

    lambdas = parse_float_list(args.lambdas)
    taus = parse_float_list(args.taus)
    frames = []
    skipped = []
    for output_dir in output_dirs:
        try:
            frames.append(evaluate_output_dir(output_dir, lambdas, taus))
            print(f'evaluated: {output_dir}')
        except (FileNotFoundError, ValueError) as exc:
            skipped.append({'output_dir': output_dir, 'reason': str(exc)})
            print(f'skipped: {output_dir} ({exc})')

    if not frames:
        raise FileNotFoundError('No valid output directories were evaluated.')
    all_results = pd.concat(frames, ignore_index=True)
    aggregate_results(all_results, args.aggregate_csv)
    if skipped:
        pd.DataFrame(skipped).to_csv(args.aggregate_csv.replace('.csv', '_skipped.csv'), index=False)
    print(f'saved aggregate summary: {args.aggregate_csv}')


if __name__ == '__main__':
    main()
