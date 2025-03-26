import csv

def convert_reaction_data_to_cypher(data, output_file):
  cypher_statements = []
  for row in data:
    # Create node
    node_id = row['id']
    create_statement = f'CREATE (n:Reaction {{id: "{node_id}"}}) RETURN n ;'
    cypher_statements.append(create_statement)

    # Set node properties
    set_properties = f'MATCH (n:Reaction {{id: "{node_id}"}}) SET '
    properties = []

    for key, value in row.items():
      if key == 'name':
        # Handle lists and strings with potential quotes
        if isinstance(value, list):
          value_str = str(value).replace("'", '"')
        else:
          value_str = repr(value)
        properties.append(f'n.{key} = {value_str}')

      if key == 'old_id':
        # Handle lists and strings with potential quotes
        if isinstance(value, list):
          value_str = str(value).replace("'", '"')
        else:
          value_str = repr(value)
        properties.append(f'n.{key} = {value_str}')

      if key == 'reaction_eq_annotation':
        # Handle lists and strings with potential quotes
        if isinstance(value, list):
          value_str = str(value).replace("'", '"')
        else:
          value_str = repr(value)
        properties.append(f'n.{key} = {value_str}')

      if key == 'reaction_eq':
        # Handle lists and strings with potential quotes
        if isinstance(value, list):
          value_str = str(value).replace("'", '"')
        else:
          value_str = repr(value)
        properties.append(f'n.{key} = {value_str}')

      if key == 'reaction_abbreviation':
        # Handle lists and strings with potential quotes
        if isinstance(value, list):
          value_str = str(value).replace("'", '"')
        else:
          value_str = repr(value)
        properties.append(f'n.{key} = {value_str}')

      if key == 'reaction_eq_name':
        # Handle lists and strings with potential quotes
        if isinstance(value, list):
          value_str = str(value).replace("'", '"')
        else:
          value_str = repr(value)
        properties.append(f'n.{key} = {value_str}')

      if key == 'GPR':
        value_GPR1 = str(value).replace("or", ',')
        value_GPR = str(value_GPR1).replace("and", ',')

        items = value_GPR.split(",")
        trimmed_items = [item.strip() for item in items]

        unique_items_GPR = set(trimmed_items)
        trimmed_items_GPR = list(unique_items_GPR)

        print(trimmed_items_GPR)

        if len(trimmed_items_GPR)==0:
          properties.append(f'n.{key} = '' ')

        if len(trimmed_items_GPR)==1:
          item0 = trimmed_items_GPR[0]
          value_str = repr(item0)
          properties.append(f'n.{key} = {value_str}')
        else:
          properties.append(f'n.{key} = {trimmed_items_GPR}')

        # Handle lists and strings with potential quotes
        # if isinstance(output_list, list):
        #   value_str = str(value).replace("'", '"')
        # else:
        #   value_str = repr(value)
        # properties.append(f'n.{key} = {value_str}')

      if key == 'lower_bound':
        # Handle lists and strings with potential quotes
        properties.append(f'n.{key} = {value}')

      if key == 'upper_bound':
        # Handle lists and strings with potential quotes
        properties.append(f'n.{key} = {value}')

      if key == 'EC_number':
        # Handle lists and strings with potential quotes
        value_EC = str(value).replace("|", ',')
        items_EC = value_EC.split(",")
        trimmed_items_EC = [item.strip() for item in items_EC]
        print(trimmed_items_EC)

        trimmed_items_EC_set = set(trimmed_items_EC)
        trimmed_items_EC_list = list(trimmed_items_EC_set)

        if len(trimmed_items_EC_list)==0:
          properties.append(f'n.{key} = '' ')

        if len(trimmed_items_EC_list)==1:
          item0 = trimmed_items_EC_list[0]
          value_str = repr(item0)
          properties.append(f'n.{key} = {value_str}')
        else:
          properties.append(f'n.{key} = {trimmed_items_EC_list}')

      if key == 'Unique_EC':
        # Handle lists and strings with potential quotes
        value_UEC = str(value).replace("|", ',')
        items_UEC = value_UEC.split(",")

        trimmed_items_UEC = [item.strip() for item in items_UEC]
        print(trimmed_items_UEC)

        trimmed_items_UEC_set = set(trimmed_items_UEC)
        trimmed_items_UEC_list = list(trimmed_items_UEC_set)

        if len(trimmed_items_UEC_list)==0:
          properties.append(f'n.{key} = '' ')

        if len(trimmed_items_UEC_list)==1:
          itemU0 = trimmed_items_UEC_list[0]
          value_str = repr(itemU0)
          properties.append(f'n.{key} = {value_str}')
        else:
          properties.append(f'n.{key} = {trimmed_items_UEC_list}')

      if key == 'Subsystem':
        # Handle lists and strings with potential quotes
        if isinstance(value, list):
          value_str = str(value).replace("'", '"')
        else:
          value_str = repr(value)
        properties.append(f'n.{key} = {value_str}')

      if key == 'KEGG_pathway':
        # Handle lists and strings with potential quotes
        if isinstance(value, list):
          value_str = str(value).replace("'", '"')
        else:
          value_str = repr(value)
        properties.append(f'n.{key} = {value_str}')

      if key == 'KEGG_iZM516_pathway':
        # Handle lists and strings with potential quotes
        if isinstance(value, list):
          value_str = str(value).replace("'", '"')
        else:
          value_str = repr(value)
        properties.append(f'n.{key} = {value_str}')

      if key == 'notes':
        # Handle lists and strings with potential quotes
        if isinstance(value, list):
          value_str = str(value).replace("'", '"')
        else:
          value_str = repr(value)
        properties.append(f'n.{key} = {value_str};')

      if key == 'annotation':
        # Handle lists and strings with potential quotes
        if isinstance(value, list):
          value_str = str(value).replace("'", '"')
        else:
          value_str1 = str(value).replace("'", '"')
          value_str = repr(value_str1)
          print(value_str)
        properties.append(f'n.{key} = {value_str}')


    set_properties += ', '.join(properties)
    cypher_statements.append(set_properties)

  # Write Cypher statements to file
  with open(output_file, 'w') as cypher_file:
    for statement in cypher_statements:
      cypher_file.write(statement + '\n')

# Example usage
data_file = '../datas/reactions.csv'  # Replace with your CSV file path
output_file = 'cypher_reactions.txt'  # Replace with your desired output file path

data = []
with open(data_file, 'r') as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    data.append(row)

convert_reaction_data_to_cypher(data, output_file)

print(f'Cypher statements written to: {output_file}')
