import argparse
import glob
import os
from itertools import combinations

import pandas as pd


def parse_views(text):
    return [item.strip() for item in str(text).split(',') if item.strip()]


def safe_acc(pred, y):
    if len(y) == 0:
        return None
    return float((pred == y).mean())


def ratio(mask, denom):
    return float(mask.sum() / max(int(denom), 1))


def agreement_rows_for_output(output_dir, views):
    pred_path = os.path.join(output_dir, 'target_predictions.csv')
    if not os.path.exists(pred_path):
        raise FileNotFoundError(pred_path)

    df = pd.read_csv(pred_path)
    pred_cols = [f'pred_{view}_idx' for view in views]
    missing = [col for col in pred_cols if col not in df.columns]
    if missing:
        raise ValueError(f'Missing prediction columns in {pred_path}: {missing}')
    if 'y_true_idx' not in df.columns:
        raise ValueError(f'Missing y_true_idx in {pred_path}')

    dataset = os.path.basename(os.path.normpath(output_dir))
    preds = df[pred_cols].astype(int)
    y = df['y_true_idx'].fillna(-1).astype(int)
    known = y >= 0
    num_samples = len(df)
    num_known = int(known.sum())

    rows = []

    all_agree = preds.nunique(axis=1) == 1
    all_agree_known = all_agree & known
    all_agree_pred = preds.loc[all_agree_known, pred_cols[0]].to_numpy()
    rows.append({
        'dataset': dataset,
        'agreement_type': 'all_views_agree',
        'views': '+'.join(views),
        'num_samples': int(num_samples),
        'num_known': int(num_known),
        'agree_count': int(all_agree.sum()),
        'agree_known_count': int(all_agree_known.sum()),
        'agree_ratio_all_samples': ratio(all_agree, num_samples),
        'agree_ratio_known_samples': ratio(all_agree_known, num_known),
        'agree_accuracy': safe_acc(all_agree_pred, y[all_agree_known].to_numpy()),
    })

    if len(views) == 3:
        exactly_two = pd.Series(False, index=df.index)
        exactly_two_pred = pd.Series(-1, index=df.index)
        for left, right in combinations(pred_cols, 2):
            other = [col for col in pred_cols if col not in (left, right)][0]
            pair_only = (preds[left] == preds[right]) & (preds[left] != preds[other])
            exactly_two |= pair_only
            exactly_two_pred.loc[pair_only] = preds.loc[pair_only, left]
        exactly_two_known = exactly_two & known
        rows.append({
            'dataset': dataset,
            'agreement_type': 'exactly_two_views_agree',
            'views': 'any_pair_not_all_three',
            'num_samples': int(num_samples),
            'num_known': int(num_known),
            'agree_count': int(exactly_two.sum()),
            'agree_known_count': int(exactly_two_known.sum()),
            'agree_ratio_all_samples': ratio(exactly_two, num_samples),
            'agree_ratio_known_samples': ratio(exactly_two_known, num_known),
            'agree_accuracy': safe_acc(exactly_two_pred[exactly_two_known].to_numpy(), y[exactly_two_known].to_numpy()),
        })

        at_least_two = all_agree | exactly_two
        at_least_two_pred = exactly_two_pred.copy()
        at_least_two_pred.loc[all_agree] = preds.loc[all_agree, pred_cols[0]]
        at_least_two_known = at_least_two & known
        rows.append({
            'dataset': dataset,
            'agreement_type': 'at_least_two_views_agree',
            'views': 'any_pair',
            'num_samples': int(num_samples),
            'num_known': int(num_known),
            'agree_count': int(at_least_two.sum()),
            'agree_known_count': int(at_least_two_known.sum()),
            'agree_ratio_all_samples': ratio(at_least_two, num_samples),
            'agree_ratio_known_samples': ratio(at_least_two_known, num_known),
            'agree_accuracy': safe_acc(at_least_two_pred[at_least_two_known].to_numpy(), y[at_least_two_known].to_numpy()),
        })

    for left_view, right_view in combinations(views, 2):
        left = f'pred_{left_view}_idx'
        right = f'pred_{right_view}_idx'
        agree = preds[left] == preds[right]
        agree_known = agree & known
        rows.append({
            'dataset': dataset,
            'agreement_type': 'pair_agree',
            'views': f'{left_view}+{right_view}',
            'num_samples': int(num_samples),
            'num_known': int(num_known),
            'agree_count': int(agree.sum()),
            'agree_known_count': int(agree_known.sum()),
            'agree_ratio_all_samples': ratio(agree, num_samples),
            'agree_ratio_known_samples': ratio(agree_known, num_known),
            'agree_accuracy': safe_acc(preds.loc[agree_known, left].to_numpy(), y[agree_known].to_numpy()),
        })

    class_rows = all_views_agreement_class_rows(dataset, df, preds, y, known, all_agree, views)
    return rows, class_rows


def all_views_agreement_class_rows(dataset, df, preds, y, known, all_agree, views):
    pred_cols = [f'pred_{view}_idx' for view in views]
    pred_col = pred_cols[0]
    num_samples = len(df)
    num_known = int(known.sum())
    all_agree_known = all_agree & known
    rows = []

    if 'y_true_idx' in df.columns:
        known_classes = sorted(int(c) for c in y[known].unique())
    else:
        known_classes = []
    pred_classes = sorted(int(c) for c in preds.loc[all_agree, pred_col].unique())
    classes = sorted(set(known_classes) | set(pred_classes))

    for class_idx in classes:
        class_agree = all_agree & (preds[pred_col] == class_idx)
        class_agree_known = class_agree & known
        class_known = known & (y == class_idx)
        correct = class_agree_known & (y == class_idx)
        rows.append({
            'dataset': dataset,
            'class_idx': int(class_idx),
            'class_label': label_for_class(df, class_idx),
            'num_samples': int(num_samples),
            'num_known': int(num_known),
            'class_known_support': int(class_known.sum()),
            'all_views_agree_class_count': int(class_agree.sum()),
            'all_views_agree_class_known_count': int(class_agree_known.sum()),
            'all_views_agree_class_correct_count': int(correct.sum()),
            'class_agree_ratio_all_samples': ratio(class_agree, num_samples),
            'class_agree_ratio_known_samples': ratio(class_agree_known, num_known),
            'class_agree_ratio_within_all_agree': ratio(class_agree, all_agree.sum()),
            'class_agree_ratio_within_class_support': ratio(class_agree_known, class_known.sum()),
            'class_agree_accuracy': ratio(correct, class_agree_known.sum()) if class_agree_known.sum() > 0 else None,
        })
    return rows


def label_for_class(df, class_idx):
    if 'y_true_idx' not in df.columns or 'y_true' not in df.columns:
        return ''
    rows = df[df['y_true_idx'].fillna(-1).astype(int) == int(class_idx)]
    if rows.empty:
        return ''
    return str(rows.iloc[0]['y_true'])


def aggregate(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    summary_rows = []
    for (agreement_type, views), group in df.groupby(['agreement_type', 'views']):
        summary_rows.append({
            'agreement_type': agreement_type,
            'views': views,
            'num_datasets': int(len(group)),
            'mean_agree_ratio_known_samples': float(group['agree_ratio_known_samples'].mean()),
            'mean_agree_accuracy': float(group['agree_accuracy'].mean()),
            'total_agree_known_count': int(group['agree_known_count'].sum()),
            'total_known': int(group['num_known'].sum()),
            'micro_agree_ratio_known_samples': float(group['agree_known_count'].sum() / max(group['num_known'].sum(), 1)),
        })
    return pd.DataFrame(summary_rows)


def aggregate_class_rows(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    summary = df.groupby(['class_idx', 'class_label'], as_index=False).agg({
        'class_known_support': 'sum',
        'all_views_agree_class_known_count': 'sum',
        'all_views_agree_class_correct_count': 'sum',
        'class_agree_ratio_known_samples': 'mean',
        'class_agree_ratio_within_class_support': 'mean',
        'class_agree_accuracy': 'mean',
    })
    summary = summary.rename(columns={
        'class_agree_ratio_known_samples': 'mean_class_agree_ratio_known_samples',
        'class_agree_ratio_within_class_support': 'mean_class_agree_ratio_within_class_support',
        'class_agree_accuracy': 'mean_class_agree_accuracy',
    })
    summary['micro_class_agree_ratio_within_class_support'] = (
        summary['all_views_agree_class_known_count']
        / summary['class_known_support'].clip(lower=1)
    )
    summary['micro_class_agree_accuracy'] = (
        summary['all_views_agree_class_correct_count']
        / summary['all_views_agree_class_known_count'].clip(lower=1)
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dirs', nargs='*', default=[])
    parser.add_argument('--glob', default=None)
    parser.add_argument('--views', default='ps,byte,mm')
    parser.add_argument('--output_csv', default='view_agreement_diagnostics.csv')
    args = parser.parse_args()

    output_dirs = list(args.output_dirs)
    if args.glob:
        output_dirs.extend(glob.glob(args.glob, recursive=True))
    output_dirs = sorted(set(output_dirs))
    if not output_dirs:
        raise ValueError('Pass --output_dirs and/or --glob.')

    views = parse_views(args.views)
    rows = []
    class_rows = []
    skipped = []
    for output_dir in output_dirs:
        try:
            output_rows, output_class_rows = agreement_rows_for_output(output_dir, views)
            rows.extend(output_rows)
            class_rows.extend(output_class_rows)
        except (FileNotFoundError, ValueError) as exc:
            skipped.append({'output_dir': output_dir, 'reason': str(exc)})

    if not rows:
        raise FileNotFoundError('No valid target_predictions.csv files were evaluated.')

    out = pd.DataFrame(rows)
    out.to_csv(args.output_csv, index=False)
    aggregate(out).to_csv(args.output_csv.replace('.csv', '_summary.csv'), index=False)
    class_out = pd.DataFrame(class_rows)
    class_out.to_csv(args.output_csv.replace('.csv', '_all_views_agree_by_class.csv'), index=False)
    aggregate_class_rows(class_rows).to_csv(
        args.output_csv.replace('.csv', '_all_views_agree_by_class_summary.csv'), index=False
    )
    if skipped:
        pd.DataFrame(skipped).to_csv(args.output_csv.replace('.csv', '_skipped.csv'), index=False)
    print(f'saved per-dataset agreement diagnostics: {args.output_csv}')
    print(f'saved agreement summary: {args.output_csv.replace(".csv", "_summary.csv")}')
    print(f'saved all-views agreement by class: {args.output_csv.replace(".csv", "_all_views_agree_by_class.csv")}')


if __name__ == '__main__':
    main()
