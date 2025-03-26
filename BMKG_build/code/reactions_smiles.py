import csv
import ast


# 读取 metabolite 表并构建代谢物 ID 和 SMILES 的映射关系
def read_metabolite_csv(file_path):
    metabolite = {}
    with open(file_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            metabolite[row['id']] = row['smiles']
    return metabolite


# 处理 reaction 表，替换代谢物 ID 为 SMILES 值
def process_reaction_csv(input_file, output_file, metabolite):
    with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            reaction_eq_annotation = row['reaction_eq_annotation']
            if reaction_eq_annotation:
                # 将字符串转换为字典
                annotation_dict = ast.literal_eval(reaction_eq_annotation)

                # 替换代谢物ID为SMILES
                new_annotation_dict = {metabolite.get(k, k): v for k, v in annotation_dict.items()}
                row['reaction_eq_annotation'] = str(new_annotation_dict)

            writer.writerow(row)


# 使用示例
input_reaction_file = '../datas/processed_reactions.csv'
output_reaction_file = '../datas/reactions_smiles.csv'
input_metabolite_file = '../datas/metabolite_smiles.csv'

# 读取代谢物表
metabolite_data = read_metabolite_csv(input_metabolite_file)

# 处理反应表
process_reaction_csv(input_reaction_file, output_reaction_file, metabolite_data)