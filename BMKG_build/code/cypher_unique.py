def remove_duplicates(input_file, output_file):
    """
    从输入文件中删除重复行并写入输出文件。

    Args:
        input_file (str): 输入文件的路径。
        output_file (str): 输出文件的路径。
    """

    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 使用集合来存储唯一行
    unique_lines = set()

    # 过滤重复行
    filtered_lines = []
    for line in lines:
        if line not in unique_lines:
            unique_lines.add(line)
            filtered_lines.append(line)

    # 将过滤后的内容写入输出文件
    with open(output_file, 'w') as f:
        f.writelines(filtered_lines)


# 示例用法
input_file = 'cypher_relationship_statements.txt'  # 输入文件路径
output_file = 'cypher_relationship_statements_unique.txt'  # 输出文件路径

remove_duplicates(input_file, output_file)