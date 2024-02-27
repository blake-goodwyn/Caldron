import pandas as pd
from datetime import datetime
import os

def combine_recipe_csvs(folder_path):
    combined_df = pd.DataFrame()
    url_set = set()

    for filename in os.listdir(folder_path):
        if filename.startswith("recipes") and filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            temp_df = pd.read_csv(file_path)

            # Filter out duplicates based on URL
            temp_df = temp_df[~temp_df['URL'].isin(url_set)]
            url_set.update(temp_df['URL'])

            combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    current_time = datetime.now().strftime("%Y-%m-%d-%H%M")
    combined_filename = f"banana-bread-combined-recipes-{current_time}.csv"

    combined_df.to_csv(os.path.join(folder_path, combined_filename), index=False)

# Example usage
combine_recipe_csvs('C:/Users/blake/Documents/GitHub/ebakery/data')
