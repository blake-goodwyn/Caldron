from dataset_linter import lint_dataset
import pandas as pd

file = 'data/bread/processed-bread-recipe-2024-03-07-1139.csv'
df = pd.read_csv(file)
df = lint_dataset(df, _DEBUG=True)
df.to_csv(file, index=False)