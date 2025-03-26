import csv

def convert_data_to_cypher(data, output_file):
  cypher_statements = []
  for row in data:
    # Create node
    node_name = row['name']
    create_statement = f'CREATE (n:Reaction {{name: "{node_name}"}}) RETURN n'
    cypher_statements.append(create_statement)

    # Set node properties
    set_properties = f'MATCH (n:Reaction {{name: "{node_name}"}}) SET '
    properties = []
    for key, value in row.items():
      if key != 'name':
        # Handle lists and strings with potential quotes
        if isinstance(value, list):
          value_str = str(value).replace("'", '"')
        else:
          value_str = repr(value)
        properties.append(f'n.{key} = {value_str}')
    set_properties += ', '.join(properties)
    cypher_statements.append(set_properties)

  # Write Cypher statements to file
  with open(output_file, 'w') as cypher_file:
    for statement in cypher_statements:
      cypher_file.write(statement + '\n')

# Example usage
data_file = 'datas/reactions1.csv'  # Replace with your CSV file path
output_file = 'cypher_statements.txt'  # Replace with your desired output file path

data = []
with open(data_file, 'r') as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    data.append(row)

convert_data_to_cypher(data, output_file)

print(f'Cypher statements written to: {output_file}')
