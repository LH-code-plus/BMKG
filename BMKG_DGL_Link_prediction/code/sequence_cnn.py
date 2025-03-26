import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json

def smiles_to_sequence(smiles, char_dict, max_len=100):
    """将SMILES字符串转换为数字序列

    Args:
        smiles: SMILES字符串
        char_dict: 字符字典
        max_len: 最大序列长度

    Returns:
        list: 数字序列
    """
    # 处理浮点数，将小数点和指数符号也加入字符字典
    smiles = str(smiles).replace('.', ' ').replace('e', ' ')  # 将小数点和指数符号替换为空格
    tokens = smiles.split()
    seq = [char_dict.get(token, char_dict['<UNK>']) for token in tokens[:max_len]]
    seq = seq + [0] * (max_len - len(seq))
    return seq

def create_char_dict(smiles_list):
    """创建字符字典

    Args:
        smiles_list: SMILES字符串列表

    Returns:
        dict: 字符字典
    """
    chars = set()
    for smiles in smiles_list:
        for token in smiles.split():
            chars.add(token)
    chars.add('<UNK>')
    char_dict = {c: i for i, c in enumerate(chars)}
    return char_dict

# 从JSON文件中加载数据
file_path = '../data/json/Sequences.json'  # 替换为你的JSON文件路径
with open(file_path, 'r') as f:
    data = json.load(f)

# 初始化DataFrame
data_dict = {}
for ec_number, smiles_list in data.items():
    if not smiles_list:
        data_dict[ec_number] = np.zeros(128)
        continue
    
    # 创建字符字典
    char_dict = create_char_dict(smiles_list)
    
    # 将SMILES字符串转换为数字序列
    sequences = [smiles_to_sequence(smiles, char_dict) for smiles in smiles_list]
    
    # 将序列转换为torch张量
    X = torch.tensor(sequences, dtype=torch.long)
    
    # 构建CNN模型
    class CNN(nn.Module):
        def __init__(self, vocab_size, embedding_dim, output_size):
            super(CNN, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.conv1 = nn.Conv1d(embedding_dim, 32, kernel_size=3)
            self.pool = nn.MaxPool1d(2)
            self.fc = nn.Linear(32 * ((max_len - 2) // 2), output_size)

        def forward(self, x):
            embedded = self.embedding(x).transpose(1, 2)  # 转换维度以便卷积操作
            conv_out = self.conv1(embedded)
            pooled_out = self.pool(conv_out)
            flattened_out = pooled_out.view(pooled_out.size(0), -1)
            out = self.fc(flattened_out)
            return out

    # 实例化模型
    vocab_size = len(char_dict)
    embedding_dim = 128
    max_len = 100  
    output_size = 128
    model = CNN(vocab_size, embedding_dim, output_size)
    
    # 生成向量
    with torch.no_grad():
        vectors = model(X).numpy()
    
    # 计算向量的平均值
    avg_vector = np.mean(vectors, axis=0)
    data_dict[ec_number] = avg_vector

# # 将结果保存到DataFrame并导出到CSV文件
# df = pd.DataFrame(list(data_dict.items()), columns=['EC_number', 'cnn_embedding'])
# df.to_csv('../data/csv/cnn_sequences111.csv', index=False)

# 将结果保存到DataFrame并导出到CSV文件
df = pd.DataFrame(list(data_dict.items()), columns=['EC_number', 'cnn_embedding'])

# 将向量转换为逗号分隔的字符串
df['cnn_embedding'] = df['cnn_embedding'].apply(lambda x: ','.join(map(str, x)))

df.to_csv('../data/csv/CKG_data/cnn_sequences111.csv', index=False)