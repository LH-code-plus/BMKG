import csv
import json

#MATCH(a:Reaction {name:'diphosphate_phosphohydrolase_c0'}),(b:Metabolite {id: 'cpd00001_c0'}) MERGE(b) - [: PARTICIPATES_IN]->(a);

#MATCH(a:Reaction {name:'diphosphate_phosphohydrolase_c0'}),(b:Metabolite {id: 'cpd00009_c0'}) MERGE(a) - [: HAS_PRODUCT]->(b);

#MATCH(a:Reaction {name:'diphosphate_phosphohydrolase_c0'}),(b:Gene {id: 'ZMO1507'}) MERGE(b) - [: CATALYZES]->(a);

#MATCH (r) DETACH DELETE r
#CREATE CONSTRAINT ON (a:Gene)ASSERT a.name IS UNIQUE
#DROP CONSTRAINT ON (a:Gene)ASSERT a.name IS UNIQUE

def convert_relationship_data_to_cypher(data, output_file):
  cypher_statements = []

  for row in data:
    # find node
    for key, value in row.items():

      node_id = row['id']
      find_statement = f'MATCH (a:Reaction {{id: "{node_id}"}}), '
      #cypher_statements.append(find_statement)
      #if( row['KEGG_pathway'] != ''):
      pathway_name = row['KEGG_pathway']

      if key == 'reaction_eq_annotation':  #代谢物和反应的关系， 代谢物和pathway的关系
        # Handle lists and strings with potential quotes
        value1 = str(value).replace("{", "").replace("}", "")
        key_value_pairs = convert_string_to_key_value_pairs(value1)

        for key_value in key_value_pairs:
          print(key_value)

          for key, value in key_value.items():  # Iterate over key-value pairs
            print(f"Key: {key}, Value: {value}")

            if int(value) <= 0:
              statement1 = find_statement + f'(b:Metabolite {{id: "{key}"}}) MERGE(b) - [: PARTICIPATES_IN]->(a);'
              print(statement1)
            elif int(value) > 0:
              statement1 = find_statement + f'(b:Metabolite {{id: "{key}"}}) MERGE(a) - [: HAS_PRODUCT]->(b);'
              print(statement1)

          if str(pathway_name) != '':
            find_statement_pathway = f'MATCH (a:Pathway {{name: "{pathway_name}"}}), '
            statement4 = find_statement_pathway + f'( b:Metabolite {{id: "{key}"}}) MERGE(b) - [: ANNOTATED_IN_PATHWAY]->(a);'
            cypher_statements.append(statement4)

          cypher_statements.append(statement1)

    #cypher_statements.append(value_str)

      if key == 'GPR':  #基因和pathway的关系
        value_GPR1 = str(value).replace("or", ',')
        value_GPR = str(value_GPR1).replace("and", ',')

        items = value_GPR.split(",")
        trimmed_items = [item.strip() for item in items]

        unique_items_GPR = set(trimmed_items)
        trimmed_items_GPR = list(unique_items_GPR)

        trimmed_items_GPR = [item for item in trimmed_items_GPR if item]

        print(trimmed_items_GPR)

        if len(trimmed_items_GPR) != 0:
          for GPRid in trimmed_items_GPR:
            #statement2 = find_statement + f'(b:Gene {{id: "{GPRid}"}}) MERGE(b) - [: CATALYZES]->(a);'
            #print(statement2)
            #cypher_statements.append(statement2)
            if str(pathway_name) != '':
              find_statement_pathway = f'MATCH (a:Pathway {{name: "{pathway_name}"}}), '
              statement5 = find_statement_pathway + f'( b:Gene {{id: "{GPRid}"}}) MERGE(b) - [: ANNOTATED_IN_PATHWAY]->(a);'
              cypher_statements.append(statement5)

      if key == 'KEGG_pathway': #反应和pathway关系
        pathway = str(value)
        if (pathway != ''):
          statement3 = find_statement + f'(b:Pathway {{name: "{pathway}"}}) MERGE(a) - [: ANNOTATED_IN_PATHWAY]->(b);'
          print(statement3)
          cypher_statements.append(statement3)

      if key == 'Unique_EC':   #蛋白质和pathway， 蛋白和反应的关系
        value_UEC = str(value).replace("|", ',')
        items_UEC = value_UEC.split(",")

        trimmed_items_UEC = [item.strip() for item in items_UEC if item]
        print(trimmed_items_UEC)

        trimmed_items_UEC_set = set(trimmed_items_UEC)
        trimmed_items_UEC_list = list(trimmed_items_UEC_set)

        if len(trimmed_items_UEC_list) == 0:
          continue
        else:
          for item in trimmed_items_UEC_list:
            statement2 = find_statement + f'(b:Protein {{EC_number: "{item}"}}) MERGE(b) - [: CATALYZES]->(a);'
            print(statement2)
            cypher_statements.append(statement2)

            # value_str = repr(item)
            if (pathway_name != ''):
              statement6 = find_statement_pathway + f'(b:Protein {{EC_number: "{item}"}}) MERGE(b) - [: ANNOTATED_IN_PATHWAY]->(a);'
              print(statement6)
              cypher_statements.append(statement6)


  # Write Cypher statements to file
  with open(output_file, 'w') as cypher_file:
    for statement in cypher_statements:
      cypher_file.write(statement + '\n')

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
      #print(key, value_str)

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


# Example usage
data_file = '../datas/reactions.csv'  # Replace with your CSV file path
output_file = 'cypher_relationship_statements.txt'  # Replace with your desired output file path

data = []
with open(data_file, 'r') as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    data.append(row)

convert_relationship_data_to_cypher(data, output_file)

print(f'Cypher statements written to: {output_file}')
