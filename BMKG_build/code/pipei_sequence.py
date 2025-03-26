import pandas as pd
import json

def match_sequences(csv_file, json_file):
    """
    根据 EC number 在 JSON 文件中匹配序列。

    Args:
        csv_file (str): CSV 文件路径。
        json_file (str): JSON 文件路径。

    Returns:
        pandas.DataFrame: 匹配结果的 DataFrame。
    """

    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 加载 JSON 数据
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 匹配并保存结果
    results = []
    for index, row in df.iterrows():
        ec_number = row['EC_number']
        for item in data:
            if item['ECNumber'] == ec_number:
                results.append({'EC_number': ec_number, 'Sequence': item['Sequence']})

    return pd.DataFrame(results)

# 示例用法
csv_file = '../datas/EC_number_Sequence.csv'
json_file = '../datas/1.json'
result_df = match_sequences(csv_file, json_file)
result_df.to_csv('../datas/matched_sequences.csv', index=False)