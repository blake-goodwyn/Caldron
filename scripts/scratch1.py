from dataset_linter import lint
import os
from recipe_state_clusters import find_state_clusters
from hmm_test import hmm_model

cwd = os.getcwd()

for file in os.listdir('data/GOOD DATASETS'):
    print("Linting: ", file)
    if file.endswith('.csv'):
        lint(os.path.abspath(file))

#find_state_clusters(file, sample=1000, max_clusters=20)
#hmm_model()