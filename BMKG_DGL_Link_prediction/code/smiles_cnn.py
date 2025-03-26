import pandas as pd
import numpy as np
import torch
import torch.nn as nn

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

def create_char_dict(df):
    """创建字符字典

    Args:
        df: 包含SMILES列的数据框

    Returns:
        dict: 字符字典
    """
    chars = set()
    for idx, smiles in enumerate(df['smiles']):
        try:
            smiles = str(smiles)  # 确保为字符串
            for token in smiles.split():  # 分离出每个token
                chars.add(token)
        except TypeError as e:
            print(f"Error: Encountered unexpected data type in smiles column at index {idx}. Error: {e}")
    chars.add('<UNK>')
    char_dict = {c: i for i, c in enumerate(chars)}
    return char_dict

# 读取CSV文件
df = pd.read_csv("../data/1Metabolite.csv")

# 创建字符字典
char_dict = create_char_dict(df)

# 将SMILES字符串转换为数字序列
sequences = [smiles_to_sequence(smiles, char_dict) for smiles in df['smiles']]
# 将序列转换为numpy数组
X = np.array(sequences)

# 构建CNN模型
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_size):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 一维卷积层示例
        self.conv1 = nn.Conv1d(embedding_dim, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        # 全连接层
        # 假设经过卷积和池化后，特征图尺寸为 32 * 5
        self.fc = nn.Linear(100, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        x_flattened = x.view(x.size(0), -1)  # 将特征图展平
        # ... 其他前向传播过程 ...
        x_flattened = x_flattened.float()
        out = self.fc(x_flattened)
        return out

# 实例化模型
vocab_size = len(char_dict)
embedding_dim = 128
output_size = 20  # 设置输出维度为20
model = CNN(vocab_size, embedding_dim, output_size)

# 加载预训练模型或训练模型
# ...

# 生成向量
with torch.no_grad():
    vectors = model(torch.tensor(X, dtype=torch.long)).numpy()

# 将向量添加到原始数据框
df['vector'] = vectors.tolist()

# 保存结果
df.to_csv("../data/1new_Metabolite.csv", index=False)
