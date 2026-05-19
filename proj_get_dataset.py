from zat.log_to_dataframe import LogToDataFrame
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import os
import argparse

info = ['ts','id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 'service']
conn_features = ['duration', 'orig_pkts', 'resp_pkts', 'orig_bytes', 'resp_bytes', 'conn_state']
ps_features = ['up', 'down', 'ps']
raw_features = ['fwd_raw', 'bwd_raw']

log2df = LogToDataFrame()



min_pkts = 5

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

    if not os.path.exists(output):
        os.mkdir(output)

    df = pd.DataFrame()
    for label in os.listdir(log_dir):
        print(label)
        # for filename in tqdm(os.listdir(os.path.join(rootdir, label))):
        for filename in os.listdir(os.path.join(log_dir, label)):
        #     print('\t', filename, end='\t')
            log_path = os.path.join(log_dir, label, filename)
            conn = log2df.create_dataframe(os.path.join(log_path, 'conn.log'), ts_index=False).set_index('uid')
            ps = log2df.create_dataframe(os.path.join(log_path, 'ps.log'), ts_index=False).set_index('uid')
            raw = log2df.create_dataframe(os.path.join(log_path, 'raw.log'), ts_index=False).set_index('uid')
            index = list(set(conn.index) & set(ps.index) & set(raw.index))
            df_tmp = pd.concat(
                [conn.loc[index][info + conn_features], ps.loc[index][ps_features], raw.loc[index][raw_features]],
                axis=1)
            df_tmp['label'] = label
            df = pd.concat([df, df_tmp])
    print(pd.DataFrame([df['label'].value_counts(), df[df['up']+df['down']>=min_pkts]['label'].value_counts()]).T)
    df.to_csv(os.path.join(output, "dataset.csv.gz"), compression='gzip')

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

    df_train.to_csv(os.path.join(output, "train.csv.gz"), compression='gzip')
    df_val.to_csv(os.path.join(output, "val.csv.gz"), compression='gzip')
    df_test.to_csv(os.path.join(output, "test.csv.gz"), compression='gzip')

    print("Split sizes:")
    print("train", len(df_train), df_train['label'].value_counts().to_dict())
    print("val", len(df_val), df_val['label'].value_counts().to_dict())
    print("test", len(df_test), df_test['label'].value_counts().to_dict())

