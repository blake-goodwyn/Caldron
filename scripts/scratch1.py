from extract_and_clean import update_ingredient_counter
import pandas as pd

# Example DataFrame
file = 'C:/Users/blake/Documents/GitHub/ebakery/data/processed-banana-bread-recipes-2024-02-27-1547.csv'
df = pd.read_csv(file)

# Specify the column to check for NaN
column_name = 'Processed_Ingredients'

# Loop through each row and check for NaN in the specified column
for index, row in df.iterrows():
    if pd.isna(row[column_name]):
        
        print(row['ID'])