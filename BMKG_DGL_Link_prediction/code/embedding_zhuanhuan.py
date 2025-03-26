import pandas as pd

def split_embedding_and_rename(df, prefix='cnnVec'):
    """
    将embedding列拆分为多个列，并自定义列名

    Args:
        df: 输入的DataFrame
        prefix: 新列名的前缀

    Returns:
        处理后的DataFrame
    """

    # 将embedding列转换为列表形式
    df['embedding'] = df['embedding'].apply(eval)

    # 获取embedding向量的维度
    embedding_dim = len(df['embedding'][0])

    # 创建新的列名列表
    new_cols = [f"{prefix}{i+1}" for i in range(embedding_dim)]

    # 使用DataFrame的构造函数创建新的DataFrame
    df_new = pd.DataFrame(df['embedding'].tolist(), columns=new_cols)

    # 将新的列添加到原始DataFrame中
    df = pd.concat([df.drop('embedding', axis=1), df_new], axis=1)

    return df

# 读取CSV文件
df = pd.read_csv('../data/csv/CKG_data/cnn_sequences111.csv')

# 调用函数处理数据
df = split_embedding_and_rename(df, prefix='cnnVec')

# 将处理后的DataFrame写入新的CSV文件
df.to_csv('../data/csv/CKG_data/new_cnn_sequences.csv', index=False)