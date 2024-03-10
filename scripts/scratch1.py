#from dataset_linter import lint, get_legible
#import os
#import pandas as pd
from recipe_collector import recipe_collect
#from bipartite import bipartite
#from recipe_state_clusters import find_state_clusters
#from hmm_test import hmm_model

#C = 0

#bipartite('data/GOOD DATASETS/processed-banana_bread-recipes-2024-02-27-1547.csv')


recipe_collect(["brownie", "biscuit", "doughnut"], 
               "C:/Users/blake/Documents/GitHub/ebakery/data", 3000) #roughly 60% result in processed recipes

# Clean all the good datasets
#for file in os.listdir('data/GOOD DATASETS'):
#    print("Linting: ", file)
#    if file.endswith('.csv'):
#        C += (len(pd.read_csv(os.path.join(os.path.dirname(file),'data\\GOOD DATASETS', file))))

#print(C)

#for file in os.listdir('data/GOOD DATASETS'):
#    if file.endswith('.csv'):
#        print(f"{file}")
#        df = pd.read_csv(os.path.join(os.path.dirname(file),'data\\GOOD DATASETS', file))
#        get_legible(df)

#find_state_clusters(file, sample=1000, max_clusters=20)
#hmm_model()