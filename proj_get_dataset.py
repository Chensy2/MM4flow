from contextlib import contextmanager
from zat.log_to_dataframe import LogToDataFrame
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import os
import gzip
import shutil
import tempfile
import argparse


info = ['ts', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 'service']
conn_features = ['duration', 'orig_pkts', 'resp_pkts', 'orig_bytes', 'resp_bytes', 'conn_state']
ps_features = ['up', 'down', 'ps']
raw_features = ['fwd_raw', 'bwd_raw']

log2df = LogToDataFrame()
min_pkts = 5


@contextmanager
def readable_log_file(log_path, name, tmp_dir=None):
    plain = os.path.join(log_path, name + ".log")
    gz = plain + ".gz"

    if os.path.exists(plain):
        yield plain
        return

    if not os.path.exists(gz):
        raise FileNotFoundError(f"Missing {plain} or {gz}")

    tmp = tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=f"_{name}.log",
        dir=tmp_dir,
        delete=False,
    )
    tmp_path = tmp.name

    try:
        with gzip.open(gz, "rb") as src, tmp:
            shutil.copyfileobj(src, tmp)
        yield tmp_path
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def split_dataset(df, train_ratio, val_ratio, test_ratio, seed):
    try:
        df_train, df_tmp = train_test_split(
            df,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True,
            stratify=df['label']
        )
        val_fraction = val_ratio / (val_ratio + test_ratio)
        df_val, df_test = train_test_split(
            df_tmp,
            train_size=val_fraction,
            random_state=seed,
            shuffle=True,
            stratify=df_tmp['label']
        )
    except ValueError as e:
        print(f"Stratified split failed: {e}")
        print("Fallback to random split without stratification.")
        df_train, df_tmp = train_test_split(
            df,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True
        )
        val_fraction = val_ratio / (val_ratio + test_ratio)
        df_val, df_test = train_test_split(
            df_tmp,
            train_size=val_fraction,
            random_state=seed,
            shuffle=True
        )

    return df_train, df_val, df_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', '-D', type=str, default='log_dir')
    parser.add_argument('--output', '-o', type=str, default='output')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    log_dir = args.log_dir
    output = args.output
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    seed = args.seed

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratio_sum}")
    if val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("val_ratio and test_ratio must both be greater than 0")

    os.makedirs(output, exist_ok=True)

    out_csv = os.path.join(output, "dataset.csv.gz")
    if os.path.exists(out_csv):
        os.remove(out_csv)

    wrote_header = False
    total_rows = 0
    label_counts = {}
    label_counts_min_pkts = {}

    labels = sorted(os.listdir(log_dir))
    for label in tqdm(labels, desc="labels", unit="label"):
        label_dir = os.path.join(log_dir, label)
        if not os.path.isdir(label_dir):
            continue

        filenames = sorted(os.listdir(label_dir))
        for filename in tqdm(filenames, desc=f"label={label}", unit="pcap", leave=False):
            log_path = os.path.join(label_dir, filename)
            if not os.path.isdir(log_path):
                continue

            try:
                with readable_log_file(log_path, "conn") as conn_file, \
                     readable_log_file(log_path, "ps") as ps_file, \
                     readable_log_file(log_path, "raw") as raw_file:

                    conn = log2df.create_dataframe(conn_file, ts_index=False).set_index("uid")
                    ps = log2df.create_dataframe(ps_file, ts_index=False).set_index("uid")
                    raw = log2df.create_dataframe(raw_file, ts_index=False).set_index("uid")

                index = list(set(conn.index) & set(ps.index) & set(raw.index))
                if not index:
                    continue

                df_tmp = pd.concat(
                    [
                        conn.loc[index][info + conn_features],
                        ps.loc[index][ps_features],
                        raw.loc[index][raw_features],
                    ],
                    axis=1,
                )
                df_tmp["label"] = label

                df_tmp.to_csv(
                    out_csv,
                    mode="a",
                    header=not wrote_header,
                    compression="gzip",
                )
                wrote_header = True

                n = len(df_tmp)
                total_rows += n
                label_counts[label] = label_counts.get(label, 0) + n
                n_min_pkts = int((df_tmp['up'] + df_tmp['down'] >= min_pkts).sum())
                label_counts_min_pkts[label] = label_counts_min_pkts.get(label, 0) + n_min_pkts

            except FileNotFoundError as e:
                tqdm.write(f"[skip-missing] {log_path}: {e}")
                continue
            except Exception as e:
                tqdm.write(f"[skip-error] {log_path}: {type(e).__name__}: {e}")
                continue

    print("total_rows:", total_rows)
    print(pd.DataFrame([pd.Series(label_counts), pd.Series(label_counts_min_pkts)], index=['all', f'pkts>={min_pkts}']).T)
    print("saved:", out_csv)

    if total_rows == 0:
        raise RuntimeError("No rows were extracted; train/val/test splits were not created.")

    df = pd.read_csv(out_csv, compression='gzip', index_col=0)
    df_train, df_val, df_test = split_dataset(df, train_ratio, val_ratio, test_ratio, seed)

    df_train.to_csv(os.path.join(output, "train.csv.gz"), compression='gzip')
    df_val.to_csv(os.path.join(output, "val.csv.gz"), compression='gzip')
    df_test.to_csv(os.path.join(output, "test.csv.gz"), compression='gzip')

    print("Split sizes:")
    print("train", len(df_train), df_train['label'].value_counts().to_dict())
    print("val", len(df_val), df_val['label'].value_counts().to_dict())
    print("test", len(df_test), df_test['label'].value_counts().to_dict())
