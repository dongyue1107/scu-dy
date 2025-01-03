"""
python v3.9.16
@Project: hotpot
@File   : model.py
@Author : Yue Dong
@Date   : 2023/9/22
@Time   : 10:13
"""

import math
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

d_model = 256
d_ff = 512
d_k = d_v = 64
n_layers = 3
n_heads = 4
batch_size = 16
seq_len = 1000
learning_rate = 0.0008
hidden_size = 85
max_length = 1000
vocab_size = 88


base_dir = '/path/to/your/base/directory'  # 设置基础路径
save_path = os.path.join(base_dir, 'directory')  # 保存路径
excel_file = os.path.join(base_dir, 'data.csv')

writer = SummaryWriter(log_dir=save_path)

device = torch.device('cuda')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=max_length):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / 512))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x:[seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)

class PositionalEncode(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncode, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / 512))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x:[seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    """
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k] True is mask
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v=len_k, d_v]
        attn_mask; [batch_size, n_heads, seq_len, seq_len]
        """
        #  torch.matmul: can calculate high dimension tensor
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value which mask is True
        attn = nn.Softmax(dim=-1)(scores)

        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)  # d_k * n_heads     64 * 8
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return self.layer_norm(output + residual), attn



class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """
        inputs:[batch_size, seq_len, d_model
        """
        residual = inputs
        output = self.fc(inputs)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, seq_len]
        """
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)  # [batch_size, src_len, d_model]

        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(118, d_model) #88

        self.pos_emb = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """

        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
#



class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.encoder = Encoder().cuda()
        self.projection = nn.Linear(d_model, 115, bias=False) # 85: target_vocab_size 45

    def forward(self, enc_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        enc_logits = self.projection(enc_outputs)

        return enc_logits


#无预训练
class DoubleBert(nn.Module):
    def __init__(self):
        super(DoubleBert, self).__init__()
        self.bert1 = Bert().cuda()
        self.bert2 = Bert().cuda()

        self.bond_length_predictor = nn.Linear(115, 1)

    def forward(self, enc_inputs1, enc_inputs2):
        bert_outputs1 = self.bert1(enc_inputs1)
        bert_outputs2 = self.bert2(enc_inputs2)
        combined_bert_outputs = bert_outputs1 + bert_outputs2
        # combined_bert_outputs = torch.cat((bert_outputs1, bert_outputs2), dim=1)
        bond_length_predictions = self.bond_length_predictor(combined_bert_outputs)

        return bond_length_predictions

pretrained_bert = Bert().cuda()
pretrained_bert.load_state_dict(torch.load(os.path.join(base_dir, 'model/state_dict'), map_location='cuda:0'))

pretrained_bert.eval()

class DoubleBertPre(nn.Module):
    def __init__(self):
        super(DoubleBertPre, self).__init__()
        self.bert1 = pretrained_bert
        self.bert2 = pretrained_bert

        self.bond_length_predictor = nn.Linear(115, 1) #85

    def forward(self, enc_inputs1, enc_inputs2):
        bert_outputs1 = self.bert1(enc_inputs1)
        bert_outputs2 = self.bert2(enc_inputs2)
        combined_bert_outputs = bert_outputs1 + bert_outputs2

        # combined_bert_outputs = torch.cat((bert_outputs1, bert_outputs2), dim=1)
        bond_length_predictions = self.bond_length_predictor(combined_bert_outputs)

        return bond_length_predictions


class Dataset(Dataset):
    def __init__(self, excel_path: Path, max_len=max_length):
        self.df_encode = pd.read_csv(excel_path, usecols=[8])
        self.all_encode = pd.read_csv(excel_path, usecols=[9])

        self.bl = pd.read_csv(excel_path, usecols=[4])
        self.max_length = max_len

    def __len__(self):
        return len(self.df_encode.index)

    def __getitem__(self, index):
        encode = self.df_encode.loc[index, :]
        a_encode = self.all_encode.loc[index, :]
        bl = self.bl.loc[index, :]
        inputs = encode[0].replace("'", "")
        inputs_list = [int(x) for x in inputs.strip('[]').split(',')]
        inputs_list = [115] + inputs_list + [116]
        # 85 86
        input_ids = torch.cat(
            [torch.tensor(inputs_list), torch.zeros(self.max_length - len(inputs_list), dtype=torch.long)]).to(device)

        a_inputs = a_encode[0].replace("'", "")
        a_inputs_list = [int(x) for x in a_inputs.strip('[]').split(',')]
        a_inputs_list = [115] + a_inputs_list + [116]
        # 85 86
        a_input_ids = torch.cat(
            [torch.tensor(a_inputs_list), torch.zeros(self.max_length - len(a_inputs_list), dtype=torch.long)]).to(device)

        bond_length = bl[0]
        return input_ids, a_input_ids, bond_length

dataset = Dataset(excel_file)
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - val_size
train_indices, val_test_indices = train_test_split(
    range(dataset_size), test_size=val_size + test_size, train_size=train_size, random_state=42
)
val_indices, test_indices = train_test_split(
    val_test_indices, test_size=test_size, random_state=42
)
#test_size = dataset_size - train_size - val_size
drop_last = True
a_train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices),
                            drop_last=drop_last)
a_val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices),
                          drop_last=drop_last)
#test_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices), drop_last=drop_last)
model = DoubleBert()
# model = DoubleBertPre()
model.to(device)

criterion_key_length = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
vali_loss = []
r2_values_list = []
max_r2 = -float('inf')

metrics_df = pd.DataFrame(columns=['Epoch', 'Train R2', 'Train MAE', 'Train RMSE', 'Validation R2', 'Validation MAE', 'Validation RMSE'])

for epoch in range(300):
    train_true_values = np.array([])
    train_predicted_values = np.array([])

    model.train()

    for input, all_input, key_lengths in a_train_loader:
        optimizer.zero_grad()
        input = input.to(device)
        all_input = all_input.to(device)
        key_lengths = key_lengths.to(device).float()

        outputs = model(input, all_input)
        outputs_s = outputs.mean(dim=1).squeeze()

        loss = criterion_key_length(outputs_s, key_lengths)

        loss.backward()
        optimizer.step()

        train_true_values = np.concatenate((train_true_values, key_lengths.cpu().detach().numpy()))
        train_predicted_values = np.concatenate((train_predicted_values, outputs_s.cpu().detach().numpy()))


    model.eval()
    val_loss = 0.0
    val_true_values = np.array([])
    val_predicted_values = np.array([])

    with torch.no_grad():
        for input, all_input, key_length in a_val_loader:
            input = input.to(device)
            all_input = all_input.to(device)
            key_length = key_length.to(device)

            outputs = model(input, all_input)
            outputs_s = outputs.mean(dim=1).squeeze()

            val_true_values = np.concatenate((val_true_values, key_length.cpu().detach().numpy()))
            val_predicted_values = np.concatenate(
                (val_predicted_values, outputs_s.cpu().detach().numpy()))

            loss = criterion_key_length(outputs_s, key_length)
            val_loss += loss.item()

        val_loss /= len(val_indices)
        vali_loss.append(val_loss)
    train_R2 = r2_score(train_true_values, train_predicted_values)
    val_R2 = r2_score(val_true_values, val_predicted_values)
    train_mae = mean_absolute_error(train_true_values, train_predicted_values)
    train_rmse = mean_squared_error(train_true_values, train_predicted_values, squared=False)

    val_mae = mean_absolute_error(val_true_values, val_predicted_values)
    val_rmse = mean_squared_error(val_true_values, val_predicted_values, squared=False)


    r2_values_list.append(val_R2)
    writer.add_scalar('Validation Loss', val_loss, epoch)
    writer.add_scalar('R2', val_R2, epoch)
    print(f'{epoch+1}: {val_R2}')

    if val_R2 > max_r2:
        max_r2 = val_R2
        torch.save(model.state_dict(), save_path + '/state_dict')
        torch.save(optimizer.state_dict(), save_path + '/optimizer_state_dict')
        plt.scatter(val_true_values, val_predicted_values, s=5, color='c', label='Validation Predictions')
        plt.plot(np.arange(min(val_true_values), max(val_true_values), 0.01),
                 np.arange(min(val_true_values), max(val_true_values), 0.01), color='black', linewidth=0.5)

        plt.xlabel('Target')
        plt.ylabel('Prediction')
        plt.legend().set_visible(False)

        plt.colorbar().remove()

        plt.savefig(save_path + f'/{epoch}')

        true_values_list = val_true_values.tolist()
        predicted_values_list = val_predicted_values.tolist()

        with open(save_path + '/true_values.txt', 'w') as true_values_file:
            for value in true_values_list:
                true_values_file.write(f"{value}\n")
        with open(save_path + '/predicted_values.txt', 'w') as predicted_values_file:
            for value in predicted_values_list:
                predicted_values_file.write(f"{value}\n")
        data = {
            'true_values': true_values_list,
            'predicted_values': predicted_values_list
        }
        df = pd.DataFrame(data)
        df.to_csv(save_path + '/true_pred.csv', index=False)


    metrics_df.loc[epoch, 'Epoch'] = epoch + 1
    metrics_df.loc[epoch, 'Train R2'] = train_R2
    metrics_df.loc[epoch, 'Train MAE'] = train_mae
    metrics_df.loc[epoch, 'Train RMSE'] = train_rmse
    metrics_df.loc[epoch, 'Validation R2'] = val_R2
    metrics_df.loc[epoch, 'Validation MAE'] = val_mae
    metrics_df.loc[epoch, 'Validation RMSE'] = val_rmse

    metrics_df.to_excel(save_path + '/metrics.xlsx', index=False)

    r2_values_list = []
print(val_loss)
writer.close()