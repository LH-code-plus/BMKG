def convert_string_to_key_value_pairs(input_str):
    """将包含键值对的字符串转换为键值对列表。
    Args:
        input_str (str): 包含键值对的输入字符串。
    Returns:
        list: 表示键值对的字典列表。
    """
    key_value_pairs = []
    for pair_str in input_str.split(","):
        # 将对字符串拆分为键和值
        key, value_str = pair_str.replace('"', '').split(":")
        print(key, value_str)

        # 将值字符串转换为适当的类型（int、float 等）
        if value_str.isdigit():
            value = int(value_str)
        elif "." in value_str:
            value = float(value_str)
        else:
            value = value_str

        # 创建表示键值对的字典
        key_value_pair = {key: value}

        # 将键值对追加到列表中
        key_value_pairs.append(key_value_pair)

    return key_value_pairs


input_str = '"cpd00001_c0":-1.0,"cpd00012_c0":-1.0,"cpd00009_c0":2.0,"cpd00067_c0":1.0'

key_value_pairs = convert_string_to_key_value_pairs(input_str)
print(key_value_pairs)