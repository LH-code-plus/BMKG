import pandas as pd

# Read CSV files
metabolite_df = pd.read_csv('../datas/metabolites.csv')
metabolite_smiles_df = pd.read_csv('../datas/metabolite_smiles.csv')

# Merge DataFrames based on 'ID' and 'id'
merged_df = metabolite_df.merge(metabolite_smiles_df, left_on='ID', right_on='id', how='left')

# Select desired columns
merged_df = merged_df[['ID', 'inchikey', 'smiles']]

# Rename columns (optional, if needed)
merged_df.columns = ['ID', 'inchikey', 'smiles']

# Write merged data to a new CSV file
merged_df.to_csv('../datas/inchikey_smile.csv', index=False)
