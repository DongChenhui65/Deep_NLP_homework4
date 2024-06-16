# -*- coding: utf-8 -*-
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
import random


def split_words_by_length(words, length):
    return [' '.join(words[i:i + length]) for i in range(0, len(words), length)]


# 读取数据
with open('.//data//corpus_sentence.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 将所有行合并成一个列表，并去掉每行的换行符
words = ' '.join(line.strip() for line in lines).split()


# 创建词汇表
tokenizer = {}
index_tokenizer = {}
word_count = {}

for word in words:
    if word not in tokenizer:
        index = len(tokenizer) + 1
        tokenizer[word] = index
        index_tokenizer[index] = word
        word_count[word] = 1
    else:
        word_count[word] += 1

vocab_size = len(tokenizer) + 1  # 0 用于填充
# if max([len(line.split()) for line in cleaned_lines]) < 100:
#     max_len = max([len(line.split()) for line in cleaned_lines])
# else:
#     max_len = 100
max_len = 10


# 转换文本为序列
def text_to_sequence(text):
    return [tokenizer[word] for word in text]


sequences = text_to_sequence(words)


def sliding_window(sequences, window_size, step=1):
    # 生成滑动窗口的句子
    return [sequences[i:i + window_size] for i in range(0, len(sequences) - window_size + 1, step)]


# 窗口大小
window_size = 11

# 生成滑动窗口的句子，步长为1
windows = sliding_window(sequences, window_size)


# 数据集类
class TextDataset(Dataset):
    def __init__(self, sequences, max_len):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # input_seq = self.sequences[idx]
        # if idx + 1 < len(self.sequences):
        #     target_seq = self.sequences[idx + 1]
        # else:
        #     target_seq = self.sequences[idx]
        seq = self.sequences[idx]
        # lenth = int(len(seq) / 20)
        lenth = 1
        input_seq = seq[:-lenth]
        target_seq = seq[lenth:]
        input_seq = input_seq + [0] * (self.max_len - len(input_seq))
        target_seq = target_seq + [0] * (self.max_len - len(target_seq))
        return torch.tensor(input_seq), torch.tensor(target_seq)


# 数据加载
dataset = TextDataset(windows, max_len)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else top1

        return outputs


# 模型超参数
INPUT_DIM = vocab_size
OUTPUT_DIM = vocab_size
ENC_EMB_DIM = 200
DEC_EMB_DIM = 200
HID_DIM = 512
N_LAYERS = 3
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=0)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(iterator):
        src = src.transpose(0, 1).to(device)
        trg = trg.transpose(0, 1).to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        output_dim = output.shape[-1]

        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src = src.transpose(0, 1).to(device)
            trg = trg.transpose(0, 1).to(device)

            output = model(src, trg, 0)

            output_dim = output.shape[-1]

            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


N_EPOCHS = 10
CLIP = 1

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    val_loss = evaluate(model, val_loader, criterion)
    print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f}')

torch.save(model.state_dict(), 'seq2seq_model.pth')

# import pickle
#
# # 加载数据集（如果需要）
# with open('train_dataset.pkl', 'rb') as f:
#     train_dataset = pickle.load(f)
# with open('val_dataset.pkl', 'rb') as f:
#     val_dataset = pickle.load(f)

# 使用时加载模型
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

# 加载模型参数
model.load_state_dict(torch.load('seq2seq_model.pth'))
model.eval()  # 设置模型为评估模式


def generate_text(model, tokenizer, text, max_len=100):
    model.eval()
    result = text
    windows = text
    for _ in range(max_len):
        tokens = []
        for x in windows:
            tokens.append(tokenizer[x])
        tokens = torch.tensor(tokens).unsqueeze(1).to(device)
        hidden, cell = model.encoder(tokens)
        input = tokens[-1].flatten()
        output, hidden, cell = model.decoder(input, hidden, cell)
        top1 = output.argmax(1)
        word = index_tokenizer[top1.item()]
        result += word
        windows += word
        windows = windows[1:]
        if word == '<eos>':
            break
    return result


# 示例生成
start_text = "马背上伏的是个高瘦的汉子，汉子手里拿了一把长剑，剑长三尺"
generated_text = generate_text(model, tokenizer, start_text)
print(generated_text)
