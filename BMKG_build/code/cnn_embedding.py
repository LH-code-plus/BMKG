import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# 定义数据集类
class SmilesDataset(Dataset):
    def __init__(self, smiles, char_dict, max_len):
        self.smiles = smiles
        self.char_dict = char_dict
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        sequence = smiles_to_sequence(smiles, self.char_dict, self.max_len)
        return torch.tensor(sequence, dtype=torch.long)

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_size):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embedding_dim, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 24, 128)  # 假设经过卷积和池化后，特征图尺寸为 32 * 24
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        conv_out = self.conv1(embedded.permute(0, 2, 1))  # 修改输入形状
        pooled = self.pool(conv_out)
        flattened = pooled.view(pooled.size(0), -1)
        x = self.dropout(flattened)
        x = self.fc1(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out

# 读取数据
df = pd.read_csv("../data/1Metabolite.csv")

# 创建字符字典
char_dict = create_char_dict(df)

# 将SMILES字符串转换为数字序列
sequences = [smiles_to_sequence(smiles, char_dict, max_len=100) for smiles in df['smiles']]

# 创建Dataset和DataLoader
dataset = SmilesDataset(sequences, char_dict, max_len=100)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 实例化模型
vocab_size = len(char_dict)
embedding_dim = 128
output_size = 20  # 设置输出维度为20
model = CNN(vocab_size, embedding_dim, output_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 训练模型
for epoch in range(10):
    for i, data in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)  # 假设有标签数据
        loss.backward()
        optimizer.step()

# 生成向量
with torch.no_grad():
    vectors = model(torch.tensor(sequences, dtype=torch.long)).numpy()
    df['vector'] = vectors.tolist()

# 保存结果
df.to_csv("../data/1new_Metabolite.csv", index=False)