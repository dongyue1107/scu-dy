"""
python v3.7.9
@Project: hotpot
@File   : test_transformerencoder.py
@Author : Yue Dong
@Date   : 2023/7/26
@Time   : 10:13
"""
import os
import math
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

base_dir = '/path/to/your/base/directory'
model_dir = os.path.join(base_dir, 'model')
dataset_dir = os.path.join(base_dir, 'dataset')

d_model = 256
d_ff = 512
d_k = d_v = 64
n_layers = 3
n_heads = 4
batch_size = 32  # 32
seq_len = 1000
learning_rate = 0.0004
hidden_size = 84

excel_path = Path(os.path.join(dataset_dir, 'CN_data', 'data.csv'))
log_dir = os.path.join(base_dir, 'pre_bert', 'model')

model_state_dict_path = os.path.join(model_dir, 'state_dict')
optimizer_state_dict_path = os.path.join(model_dir, 'optimizer_state_dict')
predictions_file_path = os.path.join(model_dir, 'predictions_and_labels.csv')

writer = SummaryWriter(log_dir=log_dir)

device = torch.device('cuda')
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
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
        # input_Q : [batch_size, len_q, d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # Q: [batch_size, n_heads, len_q, d_k]
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
        # enc_input to same Q, K, V
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


class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.encoder = Encoder().cuda()
        self.projection = nn.Linear(d_model, 115, bias=False)  # 84: target_vocab_size 45

    def forward(self, enc_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        enc_logits = self.projection(enc_outputs)
        return enc_logits


class KeyLengthPredictionHead(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(KeyLengthPredictionHead, self).__init__()
        self.fc1 = nn.Linear(hidden_size, input_size)
        self.fc2 = nn.Linear(input_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.mean(x, dim=1)
        prediction = self.fc2(x)
        return prediction


class KeyNumberPredictionHead(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(KeyNumberPredictionHead, self).__init__()
        self.fc1 = nn.Linear(hidden_size, input_size)
        self.fc2 = nn.Linear(input_size, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.mean(x, dim=1)
        prediction_logits = self.fc2(x)
        prediction_probs = F.softmax(prediction_logits, dim=1)
        predicted_number = torch.argmax(prediction_probs, dim=1) + 1  # Convert to integer lengths (1-10)
        return predicted_number


class MyDataset(Dataset):
    def __init__(self, excel_path: Path):
        self.df_encode = pd.read_csv(excel_path, usecols=[2])
        self.max_length = 1000
        self.mask_prob = 0.12
        self.mask_value = 117
        self.pretend_replace_prob = 0.015

    def __len__(self):
        return len(self.df_encode.index)

    def __getitem__(self, index):
        encode = self.df_encode.loc[index, :]
        inputs = encode[0].replace("'", "")
        inputs_list = [int(x) for x in inputs.strip('[]').split(',')]
        inputs_list = [115] + inputs_list + [116]
        input_ids_nopadding = torch.tensor(inputs_list)
        input_ids = torch.cat(
            [torch.tensor(inputs_list), torch.zeros(self.max_length - len(inputs_list), dtype=torch.long)])

        mask = torch.triu(torch.ones(1000, 1000))
        mask_attention = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        mask_indices = torch.bernoulli(torch.full(input_ids_nopadding.shape, self.mask_prob)).bool()
        mask_indices[0] = False
        mask_indices[-1] = False

        input_ids_masked = input_ids_nopadding.clone()
        input_ids_masked[mask_indices] = self.mask_value

        replace_indices = torch.bernoulli(torch.full(input_ids_nopadding.shape, self.pretend_replace_prob)).bool()
        replace_indices[0] = False
        replace_indices[-1] = False

        input_ids_masked[replace_indices] = torch.randint(low=0, high=114, size=(replace_indices.sum(),),
                                                          dtype=torch.long)
        pretend_replace_mask = torch.bernoulli(torch.full(input_ids_nopadding.shape, self.pretend_replace_prob)).bool()
        pretend_replace_mask[0] = False
        pretend_replace_mask[-1] = False

        input_masked = torch.cat(
            [input_ids_masked, torch.zeros(self.max_length - len(input_ids_masked), dtype=torch.long)])
        mask_token_id = torch.zeros_like(input_ids_nopadding, dtype=torch.bool)
        mask_token_id[replace_indices | mask_indices | pretend_replace_mask] = True
        mask_token_id = torch.cat(
            [mask_token_id, torch.zeros(self.max_length - mask_token_id.size(0), dtype=torch.bool)])

        mask_token_ids = mask_token_id.view(-1)
        return input_masked, input_ids, mask_token_ids


dataset = MyDataset(excel_path)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - val_size
train_indices, val_indices = train_test_split(
    range(dataset_size), test_size=val_size, train_size=train_size, random_state=42
)

drop_last = True
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices),
                          drop_last=drop_last)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices), drop_last=drop_last)

model = Bert()
model.load_state_dict(torch.load(model_state_dict_path , map_location='cuda'))
model = model.to(device)


dummy_input = torch.randint(0, 117, (batch_size, seq_len), dtype=torch.long)
dummy_input = dummy_input.to(device)
dummy_input = dummy_input.to(device)

writer.add_graph(model, (dummy_input,))


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.load_state_dict(torch.load(optimizer_state_dict_path , map_location='cuda'))
criterion = nn.CrossEntropyLoss()

num_epochs = 1000


tra_loss = []
vali_loss = []
accu = []
tra_accu = []
prediction = []
label = []
best_accuracy = 0.0
all_predictions = []
all_labels = []

for epoch in range(458, 459):

    model.train()
    running_loss = 0.0
    tra_total_predictions = 0
    tra_correct_predictions = 0
    for input_masked, input_ids, mask_token_id in train_loader:
        optimizer.zero_grad()
        input_masked = input_masked.to(device)
        input_ids = input_ids.to(device)
        mask_token_id = mask_token_id.to(device)
        outputs = model(input_masked)
        predicted_token_logits = outputs[mask_token_id]
        input_label = input_ids[mask_token_id]

        loss = criterion(predicted_token_logits, input_label)
        _, predicted = torch.max(predicted_token_logits, 1)
        tra_total_predictions += input_label.numel()

        tra_correct_predictions += (predicted == input_label).sum().item()

        loss.backward()
        optimizer.step()


        running_loss += loss.item()

    tra_accuracy = tra_correct_predictions / tra_total_predictions
    tra_accu.append(tra_accuracy)

    train_loss = running_loss / len(train_indices)
    tra_loss.append(train_loss)

    writer.add_scalar('Training Loss', train_loss, epoch)
    writer.add_scalar('Training Accuracy', tra_accuracy, epoch)

    model.eval()
    val_loss = 0.0
    total_predictions = 0
    correct_predictions = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for input_masked, input_ids, mask_token_id in val_loader:
            input_masked = input_masked.to(device)
            input_ids = input_ids.to(device)
            mask_token_id = mask_token_id.to(device)

            outputs = model(input_masked)
            predicted_token_logits = outputs[mask_token_id]
            input_label = input_ids[mask_token_id]

            _, predicted = torch.max(predicted_token_logits, 1)


            all_predictions.extend(predicted)
            all_labels.extend(input_label)

            loss = criterion(predicted_token_logits, input_label)
            _, predicted = torch.max(predicted_token_logits, 1)

            total_predictions += input_label.numel()

            val_loss += loss.item()

            correct_predictions += (predicted == input_label).sum().item()

        val_loss /= len(val_indices)
        accuracy = correct_predictions / total_predictions

        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Accuracy', accuracy, epoch)

        vali_loss.append(val_loss)
        accu.append(accuracy)

        print(f"Epoch {epoch + 1} {train_loss:.4f}", f"{tra_accuracy:.4f}", f"{val_loss:.4f}", f"{accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), model_state_dict_path )
            torch.save(optimizer.state_dict(), optimizer_state_dict_path )

    torch.cuda.empty_cache()
df = pd.DataFrame({
    'predicted': [item for sublist in all_predictions for item in sublist],
    'label': [item for sublist in all_labels for item in sublist]
})

df.to_csv(predictions_file_path, index=False)
writer.close()

