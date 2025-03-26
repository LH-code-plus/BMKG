import pandas as pd

def convert_embedding_to_csv(df):
    """
    将DataFrame中cnn_embedding列的字符串向量转换为逗号分隔的数值

    Args:
        df: 输入的DataFrame

    Returns:
        处理后的DataFrame
    """

    def convert_to_csv(text):
        # 去除多余的空格和方括号
        text = text.strip('[]').replace(' ', '')
        # 使用逗号分隔数值
        return text.replace(' ', ',')

    df['cnn_embedding'] = df['cnn_embedding'].apply(convert_to_csv)
    return df

# 读取CSV文件
df = pd.read_csv('../data/csv/CKG_data/cnn_sequences.csv')

# 处理数据
df = convert_embedding_to_csv(df)

# 保存结果到新的CSV文件
df.to_csv('../data/csv/CKG_data/cnn_sequences_xiu.csv', index=False)
