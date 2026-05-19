from datasets import Dataset
from transformers import BertTokenizerFast
from transformers import BertConfig, BertForMaskedLM
from transformers import TrainingArguments, Trainer
from safetensors.torch import load_model
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime
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

    def set_encoder_requires_grad(self, requires_grad):
        self.ps_encoder.requires_grad_(requires_grad=requires_grad)
        self.bytes_encoder.requires_grad_(requires_grad=requires_grad)

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

    def set_encoder_requires_grad(self, requires_grad):
        self.encoder.requires_grad_(requires_grad=requires_grad)

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


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        output = model(inputs)
        loss_cls = F.cross_entropy(output['y_logit'], inputs['y_label'])
        return (loss_cls, {'y_logit': output['y_logit']}) if return_outputs else loss_cls


def compute_metrics(pred):
    y_label = pred.label_ids[-1]
    y_pred = pred.predictions
    acc_y = (y_label == y_pred).mean()
    return {'accuracy_y': acc_y}


def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"There are {gpu_count} GPUs is available.")
else:
    gpu_count = 1


min_pkts = 5
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
pre_timestamp_ps = '202406132201'
pre_timestamp_raw = '202407081419'

model_name = "MM4flow"


def preprocess_dataframe(df, modality, label2idx=None):
    if modality in ['mm', 'ps']:
        df['ps'] = df['ps'].apply(func_ps)
    if modality in ['mm', 'byte']:
        df['fwd_raw'] = df['fwd_raw'].apply(drop_empty).fillna(' ').apply(func_bytes)
        df['bwd_raw'] = df['bwd_raw'].apply(drop_empty).fillna(' ').apply(func_bytes)

    if label2idx is not None:
        df['y_label'] = df['label'].map(label2idx)
    return df


def dataset_columns(modality):
    if modality == 'ps':
        return ['ps', 'y_label']
    if modality == 'byte':
        return ['fwd_raw', 'bwd_raw', 'y_label']
    return ['fwd_raw', 'bwd_raw', 'ps', 'y_label']


def tensor_columns(modality):
    if modality == 'ps':
        return ['ps', 'ps_attention_mask', 'y_label']
    if modality == 'byte':
        return ['raw', 'raw_attention_mask', 'raw_token_type_ids', 'y_label']
    return ['ps', 'ps_attention_mask', 'raw', 'raw_attention_mask', 'raw_token_type_ids', 'y_label']


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


def make_dataset(df, modality, shuffle=True):
    dataset = Dataset.from_pandas(df[dataset_columns(modality)])
    dataset = dataset.map(make_encoder(modality), batched=True)
    remove_columns = [col for col in ['fwd_raw', 'bwd_raw', 'ps', 'uid'] if col in dataset.column_names]
    dataset = dataset.remove_columns(remove_columns)
    dataset.set_format(type='torch', columns=tensor_columns(modality))
    return dataset.shuffle() if shuffle else dataset


def load_configs(modality):
    ps_config, bytes_config = None, None
    hyperparameters_ps, hyperparameters_bytes = None, None

    if modality in ['mm', 'ps']:
        with open(os.path.join('model', f'{premodel_name_ps}_{pre_timestamp_ps}', 'hyperparameters.json')) as f:
            hyperparameters_ps = json.load(f)
        print("MM4flow-ps", hyperparameters_ps)
        ps_config = BertConfig(
            vocab_size=len(tokenizer_ps.get_vocab()), max_position_embeddings=max_length_ps,
            hidden_size=hyperparameters_ps['d_model'], num_hidden_layers=hyperparameters_ps['n_layer'],
            num_attention_heads=hyperparameters_ps['n_head'], intermediate_size=hyperparameters_ps['dim_ff'],
        )

    if modality in ['mm', 'byte']:
        with open(os.path.join('model', f'{premodel_name_raw}_{pre_timestamp_raw}', 'hyperparameters.json')) as f:
            hyperparameters_bytes = json.load(f)
        print("MM4flow-raw", hyperparameters_bytes)
        bytes_config = BertConfig(
            vocab_size=len(tokenizer_bytes.get_vocab()), max_position_embeddings=max_length_bytes,
            hidden_size=hyperparameters_bytes['d_model'], num_hidden_layers=hyperparameters_bytes['n_layer'],
            num_attention_heads=hyperparameters_bytes['n_head'], intermediate_size=hyperparameters_bytes['dim_ff'],
        )

    return ps_config, bytes_config


def build_model(modality, ps_config, bytes_config, num_classes):
    if modality == 'ps':
        model = UniModalClassifier(ps_config, num_classes=num_classes, modality='ps').to(device)
        model.weight_init(os.path.join('model', f'{premodel_name_ps}_{pre_timestamp_ps}'))
        return model
    if modality == 'byte':
        model = UniModalClassifier(bytes_config, num_classes=num_classes, modality='byte').to(device)
        model.weight_init(os.path.join('model', f'{premodel_name_raw}_{pre_timestamp_raw}'))
        return model

    model = Classifier(ps_config=ps_config, bytes_config=bytes_config, num_classes=num_classes).to(device)
    model.weight_init(
        premodel_ps_path=os.path.join('model', f'{premodel_name_ps}_{pre_timestamp_ps}'),
        premodel_bytes_path=os.path.join('model', f'{premodel_name_raw}_{pre_timestamp_raw}')
    )
    return model


def make_training_args(model_dir, learning_rate, batch_size, n_epochs, label_names, bf16):
    return TrainingArguments(
        output_dir=model_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=n_epochs,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy='epoch',
        save_strategy='no',
        label_names=label_names,
        bf16=bf16,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', '-T', type=str, default=None)
    parser.add_argument('--dataset', '-D', type=str, default='dataset')
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--n_epochs', '-ep', type=int, default=10)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-5)
    parser.add_argument('--modality', '-M', choices=['mm', 'ps', 'byte'], default='mm')
    parser.add_argument('--eval_per_epoch', '-ev', type=int, default=1)
    parser.add_argument('--save_per_epoch', '-sv', type=int, default=1)

    args = parser.parse_args()
    timestamp = args.timestamp
    dataset_path = args.dataset
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    output = args.output
    modality = args.modality

    model_root = os.path.join('model-classifier', output) if output is not None else 'model-classifier'
    if not os.path.exists(model_root):
        os.mkdir(model_root)

    df_train = pd.read_csv(f'{dataset_path}/train.csv.gz', compression='gzip', index_col=0)
    labels = list(df_train['label'].value_counts().index)
    label2idx = dict(zip(labels, range(len(labels))))
    num_classes = len(label2idx)
    df_train = preprocess_dataframe(df_train, modality, label2idx=label2idx)
    trainset = make_dataset(df_train, modality)

    df_eval = pd.read_csv(f'{dataset_path}/val.csv.gz', compression='gzip', index_col=0)
    df_eval = preprocess_dataframe(df_eval, modality, label2idx=label2idx)
    evalset = make_dataset(df_eval, modality)

    df_test = pd.read_csv(f'{dataset_path}/test.csv.gz', compression='gzip', index_col=0)
    df_test = preprocess_dataframe(df_test, modality, label2idx=label2idx)
    testset = make_dataset(df_test, modality)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
    if not os.path.exists(os.path.join(model_root, timestamp)):
        os.mkdir(os.path.join(model_root, timestamp))

    model_type = f'{modality}-finetune'
    info = {
        'dataset': dataset_path,
        'modality': modality,
        'model_type': model_type,
        'premodel_name_ps': premodel_name_ps,
        'pre_timestamp_ps': pre_timestamp_ps,
        'premodel_name_raw': premodel_name_raw,
        'pre_timestamp_raw': pre_timestamp_raw,
        'num_classes': num_classes,
        'label2idx': label2idx
    }

    ps_config, bytes_config = load_configs(modality)

    training_results = []
    model_info_dir = os.path.join(model_root, timestamp, model_name)
    if not os.path.exists(model_info_dir):
        os.mkdir(model_info_dir)
    with open(os.path.join(model_info_dir, 'info.json'), 'w') as f:
        json.dump(info, f)

    model_dir = os.path.join(model_info_dir, model_type)
    print(model_dir)

    model = build_model(modality, ps_config=ps_config, bytes_config=bytes_config, num_classes=num_classes)
    label_names = tensor_columns(modality)
    bf16 = torch.cuda.is_available()

    model.set_encoder_requires_grad(False)
    training_args = make_training_args(model_dir, learning_rate, batch_size, n_epochs, label_names, bf16)
    trainer = MyTrainer(
        model=model, args=training_args,
        train_dataset=trainset, eval_dataset=evalset,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics
    )
    trainer.train()

    model.set_encoder_requires_grad(True)
    training_args = make_training_args(model_dir, learning_rate, batch_size, n_epochs, label_names, bf16)
    trainer = MyTrainer(
        model=model, args=training_args,
        train_dataset=trainset, eval_dataset=evalset,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_state()
    torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))

    model.eval()
    y_pred = np.array([], dtype=np.int64)
    for i in tqdm(range(0, testset.num_rows, batch_size)):
        batch_inputs = {k: v.to(device) for k, v in testset[i:i + batch_size].items()}
        y_pred_tmp = model(batch_inputs)['y_logit'].argmax(1).cpu().numpy()
        y_pred = np.concatenate([y_pred, y_pred_tmp])
    print(accuracy_score(y_true=testset['y_label'], y_pred=y_pred))
    print(confusion_matrix(y_true=testset['y_label'], y_pred=y_pred))
    print(classification_report(y_true=testset['y_label'], y_pred=y_pred, digits=4, target_names=labels))

    training_result = {
        'type': f"BERT-{modality}-{model_type}",
        'model_type': model_type,
        'modality': modality,
        'accuracy': accuracy_score(y_true=testset['y_label'], y_pred=y_pred),
        'confusion_matrix': confusion_matrix(y_true=testset['y_label'], y_pred=y_pred).tolist(),
        'classification_report': classification_report(y_true=testset['y_label'], y_pred=y_pred, output_dict=True)
    }
    training_results.append(training_result)
    with open(os.path.join(model_root, timestamp, 'training_results.json'), 'w') as f:
        json.dump(training_results, f)
