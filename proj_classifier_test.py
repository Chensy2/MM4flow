from zat.log_to_dataframe import LogToDataFrame
from datasets import Dataset
from transformers import BertTokenizerFast
from transformers import BertConfig, BertForMaskedLM
from safetensors.torch import load_model
from torch import nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import torch
import json
import argparse


class Classifier(nn.Module):
    def __init__(self, ps_config, bytes_config, num_classes):
        super(Classifier, self).__init__()
        self.ps_config, self.bytes_config = ps_config, bytes_config
        self.ps_encoder = BertForMaskedLM(ps_config)
        self.bytes_encoder = BertForMaskedLM(bytes_config)
        self.ps_cross_attention = nn.MultiheadAttention(embed_dim=ps_config.hidden_size, num_heads=4, batch_first=True)
        self.bytes_cross_attention = nn.MultiheadAttention(embed_dim=bytes_config.hidden_size, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(ps_config.hidden_size + bytes_config.hidden_size, num_classes)
        )

    def weight_init(self, premodel_ps_path, premodel_bytes_path):
        load_model(self.ps_encoder, os.path.join(premodel_ps_path, 'model.safetensors'), strict=False)
        load_model(self.bytes_encoder, os.path.join(premodel_bytes_path, 'model.safetensors'), strict=False)

    def forward(self, inputs):
        r = self.forward_with_attention_weight(inputs)
        return {'y_logit': r['y_logit']}

    def forward_with_attention_weight(self, inputs):
        ps_outputs = self.ps_encoder.bert(input_ids=inputs['ps'], attention_mask=inputs['ps_attention_mask'])
        raw_outputs = self.bytes_encoder.bert(
            input_ids=inputs['raw'],
            attention_mask=inputs['raw_attention_mask'],
            token_type_ids=inputs['raw_token_type_ids']
        )

        outputs = torch.concat([ps_outputs.last_hidden_state, raw_outputs.last_hidden_state], dim=1)
        key_padding_mask = (1 - torch.concat([inputs['ps_attention_mask'], inputs['raw_attention_mask']], dim=1)).bool()
        ps_attn_output, ps_attn_output_weights = self.ps_cross_attention(
            ps_outputs.last_hidden_state, outputs, outputs, key_padding_mask=key_padding_mask
        )
        raw_attn_output, raw_attn_output_weights = self.bytes_cross_attention(
            raw_outputs.last_hidden_state, outputs, outputs, key_padding_mask=key_padding_mask
        )

        memory_ps, memory_raw = ps_attn_output[:, 0, :], raw_attn_output[:, 0, :]
        y_logit = self.classifier(torch.concat([memory_ps, memory_raw], dim=1))
        return {'y_logit': y_logit, 'ps_attn_weights': ps_attn_output_weights, 'raw_attn_weights': raw_attn_output_weights}


class UniModalClassifier(nn.Module):
    def __init__(self, config, num_classes, modality):
        super(UniModalClassifier, self).__init__()
        if modality not in ['ps', 'byte']:
            raise ValueError(f"Unsupported unimodal classifier modality: {modality}")
        self.modality = modality
        self.encoder = BertForMaskedLM(config)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def weight_init(self, premodel_path):
        load_model(self.encoder, os.path.join(premodel_path, 'model.safetensors'), strict=False)

    def forward(self, inputs):
        if self.modality == 'ps':
            outputs = self.encoder.bert(input_ids=inputs['ps'], attention_mask=inputs['ps_attention_mask'])
        else:
            outputs = self.encoder.bert(
                input_ids=inputs['raw'],
                attention_mask=inputs['raw_attention_mask'],
                token_type_ids=inputs['raw_token_type_ids']
            )
        y_logit = self.classifier(outputs.last_hidden_state[:, 0, :])
        return {'y_logit': y_logit}


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"There are {gpu_count} GPUs is available.")
else:
    gpu_count = 1


info_features = ['ts', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 'service']
conn_features = ['duration', 'orig_pkts', 'resp_pkts', 'orig_bytes', 'resp_bytes', 'conn_state']
ps_features = ['up', 'down', 'ps']
raw_features = ['fwd_raw', 'bwd_raw']

log2df = LogToDataFrame()

max_length_ps = 256
max_length_bytes = 512


def func_ps(ps):
    r = []
    for burst in ps.split(','):
        p_len, p_count = burst.split(':')
        r += [f"p{p_len}t"] * int(p_count)
        if len(r) > max_length_ps:
            break
    return ' '.join(r[0:max_length_ps])


def func_bytes(s):
    return ' '.join([s[i:i + 2] for i in range(0, len(s), 2)])


drop_empty = lambda x: x if x != '(empty)' else np.nan

tokenizer_ps = BertTokenizerFast.from_pretrained('tokenizer_bert/ps_tokenizer')
tokenizer_bytes = BertTokenizerFast.from_pretrained('tokenizer_bert/bytes_tokenizer')

premodel_name_ps = 'BERT-ps'
premodel_name_raw = 'BERT-bytes'

model_name = "MM4flow"
min_pkts = 5


def dataset_columns(modality):
    if modality == 'ps':
        return ['ps']
    if modality == 'byte':
        return ['fwd_raw', 'bwd_raw']
    return ['fwd_raw', 'bwd_raw', 'ps']


def tensor_columns(modality):
    if modality == 'ps':
        return ['ps', 'ps_attention_mask']
    if modality == 'byte':
        return ['raw', 'raw_attention_mask', 'raw_token_type_ids']
    return ['ps', 'ps_attention_mask', 'raw', 'raw_attention_mask', 'raw_token_type_ids']


def make_encoder(modality):
    def encode(examples):
        encoded = {}
        if modality in ['mm', 'ps']:
            ps = tokenizer_ps(
                examples['ps'],
                truncation=True,
                padding="max_length",
                max_length=max_length_ps,
                return_special_tokens_mask=True
            )
            encoded.update({'ps': ps['input_ids'], 'ps_attention_mask': ps['attention_mask']})
        if modality in ['mm', 'byte']:
            raw = tokenizer_bytes(
                list(zip(examples['fwd_raw'], examples['bwd_raw'])),
                truncation=True,
                padding="max_length",
                max_length=max_length_bytes,
                return_special_tokens_mask=True
            )
            encoded.update({
                'raw': raw['input_ids'],
                'raw_attention_mask': raw['attention_mask'],
                'raw_token_type_ids': raw['token_type_ids']
            })
        return encoded

    return encode


def make_dataset(df, modality):
    dataset = Dataset.from_pandas(df[dataset_columns(modality)])
    dataset = dataset.map(make_encoder(modality), batched=True)
    remove_columns = [col for col in ['fwd_raw', 'bwd_raw', 'uid', '__index_level_0__'] if col in dataset.column_names]
    dataset = dataset.remove_columns(remove_columns)
    dataset.set_format(type='torch', columns=tensor_columns(modality))
    return dataset


def load_eval_csv(eval_csv):
    if not os.path.exists(eval_csv):
        raise FileNotFoundError(f"Missing eval csv: {eval_csv}")
    return pd.read_csv(eval_csv, compression='gzip', index_col=0)


def load_logs(rootdir, modality):
    df_test = pd.DataFrame()
    for filename in tqdm(os.listdir(rootdir), desc='reading pcap_log'):
        log_path = os.path.join(rootdir, filename)
        conn = log2df.create_dataframe(os.path.join(log_path, 'conn.log'), ts_index=False).set_index('uid')

        if modality == 'ps':
            ps = log2df.create_dataframe(os.path.join(log_path, 'ps.log'), ts_index=False).set_index('uid')
            index = list(set(conn.index) & set(ps.index))
            df_tmp = pd.concat([conn.loc[index][info_features + conn_features], ps.loc[index][ps_features]], axis=1)
        elif modality == 'byte':
            raw = log2df.create_dataframe(os.path.join(log_path, 'raw.log'), ts_index=False).set_index('uid')
            index = list(set(conn.index) & set(raw.index))
            df_tmp = pd.concat([conn.loc[index][info_features + conn_features], raw.loc[index][raw_features]], axis=1)
        else:
            ps = log2df.create_dataframe(os.path.join(log_path, 'ps.log'), ts_index=False).set_index('uid')
            raw = log2df.create_dataframe(os.path.join(log_path, 'raw.log'), ts_index=False).set_index('uid')
            index = list(set(conn.index) & set(ps.index) & set(raw.index))
            df_tmp = pd.concat(
                [conn.loc[index][info_features + conn_features], ps.loc[index][ps_features], raw.loc[index][raw_features]],
                axis=1
            )
        df_test = pd.concat([df_test, df_tmp])

    return df_test


def preprocess_dataframe(df, modality):
    if modality in ['mm', 'ps']:
        df = df[df['up'] + df['down'] >= min_pkts]
        df['ps'] = df['ps'].apply(func_ps)
    if modality in ['mm', 'byte']:
        df['fwd_raw'] = df['fwd_raw'].apply(drop_empty).fillna(' ').apply(func_bytes)
        df['bwd_raw'] = df['bwd_raw'].apply(drop_empty).fillna(' ').apply(func_bytes)
    return df


def load_configs(modality, info):
    ps_config, bytes_config = None, None
    pre_timestamp_ps = info.get("pre_timestamp_ps")
    pre_timestamp_raw = info.get("pre_timestamp_raw")

    if modality in ['mm', 'ps']:
        with open(os.path.join('model', f'{premodel_name_ps}_{pre_timestamp_ps}', 'hyperparameters.json')) as f:
            hyperparameters_ps = json.load(f)
        ps_config = BertConfig(
            vocab_size=len(tokenizer_ps.get_vocab()), max_position_embeddings=max_length_ps,
            hidden_size=hyperparameters_ps['d_model'], num_hidden_layers=hyperparameters_ps['n_layer'],
            num_attention_heads=hyperparameters_ps['n_head'], intermediate_size=hyperparameters_ps['dim_ff'],
        )

    if modality in ['mm', 'byte']:
        with open(os.path.join('model', f'{premodel_name_raw}_{pre_timestamp_raw}', 'hyperparameters.json')) as f:
            hyperparameters_bytes = json.load(f)
        bytes_config = BertConfig(
            vocab_size=len(tokenizer_bytes.get_vocab()), max_position_embeddings=max_length_bytes,
            hidden_size=hyperparameters_bytes['d_model'], num_hidden_layers=hyperparameters_bytes['n_layer'],
            num_attention_heads=hyperparameters_bytes['n_head'], intermediate_size=hyperparameters_bytes['dim_ff'],
        )

    return ps_config, bytes_config


def build_model(modality, ps_config, bytes_config, num_classes, info):
    if modality == 'ps':
        return UniModalClassifier(ps_config, num_classes=num_classes, modality='ps').to(device)
    if modality == 'byte':
        return UniModalClassifier(bytes_config, num_classes=num_classes, modality='byte').to(device)

    return Classifier(ps_config=ps_config, bytes_config=bytes_config, num_classes=num_classes).to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-D', type=str, default='dataset')
    parser.add_argument('--model_ts', '-m', type=str, default='202412110000')
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--output', '-o', type=str, default='result.csv')
    parser.add_argument('--eval_csv', type=str, default=None)

    args = parser.parse_args()
    rootdir = args.dataset
    model_ts = args.model_ts
    batch_size = args.batch_size
    output = args.output
    eval_csv = args.eval_csv

    model_path = os.path.join('model-classifier', model_ts)
    with open(os.path.join(model_path, model_name, "info.json")) as f:
        info = json.load(f)
    modality = info.get("modality", "mm")

    label2idx = info["label2idx"]
    num_classes = len(label2idx)
    idx2label = dict(zip(label2idx.values(), label2idx.keys()))

    if eval_csv is not None:
        df_test = load_eval_csv(eval_csv)
    else:
        df_test = load_logs(rootdir, modality)
    df_test = preprocess_dataframe(df_test, modality)
    testset = make_dataset(df_test, modality)

    ps_config, bytes_config = load_configs(modality, info)

    model_type = info.get("model_type", f'{modality}-finetune')
    model_dir = os.path.join(model_path, model_name, model_type)
    if not os.path.exists(os.path.join(model_dir, "pytorch_model.bin")) and modality == 'mm':
        model_dir = os.path.join(model_path, model_name, 'finetune')

    model = build_model(modality, ps_config=ps_config, bytes_config=bytes_config, num_classes=num_classes, info=info)
    model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=device))
    model.eval()

    y_pred = np.array([], dtype=np.int64)
    for i in tqdm(range(0, testset.num_rows, batch_size), desc='Detection'):
        batch_inputs = {k: v.to(device) for k, v in testset[i:i + batch_size].items()}
        y_pred_tmp = model(batch_inputs)['y_logit'].argmax(1).cpu().numpy()
        y_pred = np.concatenate([y_pred, y_pred_tmp])

    df_test['pred'] = y_pred
    df_test['pred'] = df_test['pred'].map(idx2label)
    print(df_test['pred'].value_counts())

    if 'label' in df_test.columns:
        y_true = df_test['label'].map(label2idx)
        known_label_mask = y_true.notna()
        if not known_label_mask.all():
            unknown_labels = sorted(df_test.loc[~known_label_mask, 'label'].dropna().unique())
            print(f"Skip {int((~known_label_mask).sum())} rows with labels not seen during training: {unknown_labels}")

        y_true_eval = y_true[known_label_mask].astype(int)
        y_pred_eval = y_pred[known_label_mask.to_numpy()]
        labels_eval = [label for label, _ in sorted(label2idx.items(), key=lambda item: item[1])]

        print("accuracy:", accuracy_score(y_true=y_true_eval, y_pred=y_pred_eval))
        print(confusion_matrix(
            y_true=y_true_eval,
            y_pred=y_pred_eval,
            labels=[label2idx[label] for label in labels_eval]
        ))
        print(classification_report(
            y_true=y_true_eval,
            y_pred=y_pred_eval,
            digits=4,
            target_names=labels_eval,
            labels=[label2idx[label] for label in labels_eval]
        ))

    df_test[info_features + conn_features + ['pred']].to_csv(output)

    print(f"The analysis results are saved in {output}.")
