import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
from preprocessing import load_data

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
BATCH_SIZE = 128
HID_DIM = 256
N_LAYERS = 3
N_HEADS = 8
PF_DIM = 512
DROPOUT = 0.1

# 加载数据
train_iterator, valid_iterator, test_iterator, SRC, TRG = load_data(BATCH_SIZE, device)
INPUT_DIM = len(SRC)
OUTPUT_DIM = len(TRG)

# 定义模型
from models import Transformer
model = Transformer(INPUT_DIM, OUTPUT_DIM, HID_DIM, N_LAYERS, N_HEADS).to(device)

# 初始化权重
def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(init_weights)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0005)
TRG_PAD_IDX = TRG.index('<pad>')
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

# 计算时间的函数
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# 训练函数
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for src, trg in iterator:
        optimizer.zero_grad()

        output = model(src, trg[:, :-1], None)

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# 评估函数
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, trg in iterator:
            output = model(src, trg[:, :-1], None)

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# 训练过程
N_EPOCHS = 10
CLIP = 1

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
