import csv

def convert_protein_data_to_cypher(data, output_file):
  cypher_statements = []
  for row in data:
    # Create node
    EC_number = row['Unique_EC']
    # Handle lists and strings with potential quotes
    value_UEC = str(EC_number).replace("|", ',')
    items_UEC = value_UEC.split(",")

    trimmed_items_UEC = [item.strip() for item in items_UEC if item]
    print(trimmed_items_UEC)

    trimmed_items_UEC_set = set(trimmed_items_UEC)
    trimmed_items_UEC_list = list(trimmed_items_UEC_set)

    if len(trimmed_items_UEC_list) == 0:
        continue
    else:
        for item in trimmed_items_UEC_list:
            #value_str = repr(item)
            create_statement = f'CREATE (n:Protein {{EC_number: "{item}"}}) RETURN n ;'
            cypher_statements.append(create_statement)

  # Write Cypher statements to file
  with open(output_file, 'w') as cypher_file:
    for statement in cypher_statements:
      cypher_file.write(statement + '\n')

# Example usage
data_file = '../datas/reactions.csv'  # Replace with your CSV file path
output_file = 'cypher_protein.txt'  # Replace with your desired output file path

data = []
with open(data_file, 'r') as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    data.append(row)

convert_protein_data_to_cypher(data, output_file)

print(f'Cypher statements written to: {output_file}')
