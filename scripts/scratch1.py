from extract_and_clean import clean
import pandas as pd

# Example DataFrame
file = 'C:/Users/blake/Documents/GitHub/ebakery/data/processed-banana-bread-recipes-2024-02-27-1547.csv'
df = pd.read_csv(file)

# Specify the column to check for NaN
column_name = 'Processed_Ingredients'

# Loop through each row and check for NaN in the specified column
for index, row in df.iterrows():
    if pd.isna(row[column_name]):
        row[column_name] = clean(row['Ingredients'])
        try:
            eval(row[column_name])
            print(row['ID'], " | PASS | ", row[column_name])
        except Exception as e:
            print(e)
            print(row['ID'], " | FAIL | ", row[column_name])

df.to_csv(file, index=False)
