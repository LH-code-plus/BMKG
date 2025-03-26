import pandas as pd

# 读取两个CSV文件
reaction_df = pd.read_csv("../data/csv/CKG_data/new_protein.csv")
additional_info_df = pd.read_csv("../data/csv/CKG_data/new_cnn_sequences.csv")

# 根据'rid'列合并两个DataFrame
merged_df = pd.merge(reaction_df, additional_info_df, on='EC_number')

# 保存合并后的DataFrame到新的CSV文件
merged_df.to_csv("../data/csv/CKG_data/final_protein.csv", index=False)