import pandas as pd
from Bio import ExPASy
from Bio import SeqIO

def get_protein_sequences_by_ec_number(ec_number):
    """
    根据EC number查询蛋白质序列，返回一个序列列表

    Args:
        ec_number: EC number

    Returns:
        List: 包含所有匹配序列的列表
    """
    sequences = []
    try:
        handle = ExPASy.get_sprot_raw(ec_number)
        for record in SeqIO.parse(handle, "swiss"):
            sequences.append(str(record.seq))
    except Exception as e:
        print(f"Error fetching sequence for {ec_number}: {e}")
    return sequences

# 读取CSV文件
df = pd.read_csv("../datas/EC_number_Sequence.csv")

# 创建一个新的DataFrame来存储结果
result_df = pd.DataFrame(columns=["EC_number", "Sequence"])

# 遍历每个EC number，查询并存储结果
for index, row in df.iterrows():
    ec_number = row["EC_number"]
    sequences = get_protein_sequences_by_ec_number(ec_number)
    for seq in sequences:
        result_df = result_df.append({"EC_number": ec_number, "Sequence": seq}, ignore_index=True)

# 将结果写入新的CSV文件
result_df.to_csv("../datas/result.csv", index=False)