import argparse
import glob
import os

import pandas as pd


def load_summary(path):
    df = pd.read_csv(path)
    if 'setting' not in df.columns:
        setting = os.path.basename(os.path.dirname(path))
        df.insert(0, 'setting', setting)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dirs',
        nargs='*',
        default=[],
        help='Output directories produced by proj_class_evidence_fusion.py.',
    )
    parser.add_argument(
        '--glob',
        default=None,
        help='Optional glob for output directories, for example outputs/mm4flow/hspf_*.',
    )
    parser.add_argument('--output_csv', default='view_reliability_aggregate_summary.csv')
    args = parser.parse_args()

    output_dirs = list(args.output_dirs)
    if args.glob:
        output_dirs.extend(glob.glob(args.glob))
    output_dirs = sorted(set(output_dirs))
    if not output_dirs:
        raise ValueError('Pass --output_dirs and/or --glob.')

    frames = []
    missing = []
    for output_dir in output_dirs:
        path = os.path.join(output_dir, 'view_reliability_score_summary.csv')
        if not os.path.exists(path):
            missing.append(output_dir)
            continue
        frames.append(load_summary(path))
    if not frames:
        raise FileNotFoundError('No view_reliability_score_summary.csv files found.')

    all_rows = pd.concat(frames, ignore_index=True)
    summary_rows = []
    for score, group in all_rows.groupby('score'):
        spearman = group['spearman_with_target_weighted_f1'].astype(float)
        summary_rows.append({
            'score': score,
            'num_settings': int(len(group)),
            'top1_hit_rate': float(group['top1_hits_best_view'].astype(bool).mean()),
            'top2_preservation_rate': float(group['top2_contains_best_view'].astype(bool).mean()),
            'worst_bottom1_rate': float(group['worst_view_bottom1'].astype(bool).mean()),
            'worst_bottom2_rate': float(group['worst_view_bottom2'].astype(bool).mean()),
            'mean_spearman': float(spearman.mean()),
            'spearman_variance': float(spearman.var(ddof=0)),
        })

    summary = pd.DataFrame(summary_rows).sort_values('score')
    summary.to_csv(args.output_csv, index=False)
    all_rows.to_csv(args.output_csv.replace('.csv', '_per_setting.csv'), index=False)
    if missing:
        pd.DataFrame({'missing_output_dir': missing}).to_csv(
            args.output_csv.replace('.csv', '_missing.csv'), index=False
        )
    print(f'saved aggregate summary: {args.output_csv}')
    print(f'saved per-setting rows: {args.output_csv.replace(".csv", "_per_setting.csv")}')
    if missing:
        print(f'skipped {len(missing)} dirs without view_reliability_score_summary.csv')


if __name__ == '__main__':
    main()
