#from dataset_linter import lint, get_legible
import os
#import pandas as pd
#from recipe_collector import recipe_collect
from bipartite import bipartite
#from recipe_state_clusters import find_state_clusters
#from hmm_test import hmm_model
from food_data import update_from_node_graph
from tqdm import tqdm

#C = 0

#file = 'data/GOOD DATASETS/processed-brownie-recipe-2024-03-10-1046.csv'

for file in tqdm(os.listdir('data/GOOD DATASETS')):
    if file.endswith('.csv'):
        try:
            bipartite(file)
            update_from_node_graph("ingredients/search_terms.txt")
        except Exception as e:
            print(f"Error: {e}")
            continue


#recipe_collect(["doughnut"], 
#               "C:/Users/blake/Documents/GitHub/ebakery/data", 3000) #roughly 60% result in processed recipes

# Clean all the good datasets
#for file in os.listdir('data/GOOD DATASETS'):
#    print("Linting: ", file)
#    if file.endswith('.csv'):
#        lint(os.path.join(os.path.dirname(file),'data\\GOOD DATASETS', file))

#print(C)

#for file in os.listdir('data/GOOD DATASETS'):
#    if file.endswith('.csv'):
#        print(f"{file}")
#        df = pd.read_csv(os.path.join(os.path.dirname(file),'data\\GOOD DATASETS', file))
#        get_legible(df)

#find_state_clusters(file, sample=1000, max_clusters=20)
#hmm_model()