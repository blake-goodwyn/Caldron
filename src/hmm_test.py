# Develop HMM from Recipe Sequences from experimental purposes

#import required libraries
import pickle
import os
from class_defs import RecipeAction, StateCluster
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def hmm_model():
    cwd = os.getcwd()
    dir = os.path.join(cwd, 'outputs')

    #import recipe sequences, recipe actions, and state clusters
    seq_path = os.path.join(dir, 'recipe_sequences.txt')
    #actions_path= os.path.join(dir, 'recipe_actions.pkl')
    clusters_path = os.path.join(dir, 'clusters.pkl')

    sequences = []

    #load the recipe actions and state clusters
    with open(clusters_path, 'rb') as f:
        clusters = pickle.load(f)

    label_map = {}
    for cluster in clusters:
        print(cluster)
        label_map[cluster.clusterID] = cluster.label

    print("Loading recipe sequences...")
    with open(seq_path, 'r') as f:
        for line in f:
            l = list(line.replace('<', '').replace('>', '').strip().split())
            sequences.append(np.array(l,dtype=int))

    padded_sequences = pad_sequences(sequences, padding='post')

    print("Splitting data into training and testing sets...")
    train_data, test_data = train_test_split(padded_sequences, test_size=0.3)

    # Your existing code to compute bic_scores
    models = []
    state_range = range(7, 10)
    for n_states in state_range:
        try:
            print(f"Fitting model with {n_states} states")
            # Create a Gaussian HMM and fit it to the training data
            model = hmm.CategoricalHMM(n_components=n_states)
            model.fit(train_data)

            # Calculate BIC and store it
            models.append([model, model.bic(test_data)])
        except Exception as e:
            print(f"Error: {e}")
            model.append([None, 0])

    # Choose the number of states with the lowest BIC
    optimal_states = state_range[np.argmin([x[1] for x in models])]
    optimal_model = models[np.argmin([x[1] for x in models])][0]
    emission_probs = optimal_model.emissionprob_

    print("Optimal number of states:", optimal_states)

    # Assuming 'emission_probs' is your emission probability matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(emission_probs, annot=False, cmap='YlGnBu')
    plt.xlabel('Observable States')
    plt.xticks(range(0,len(label_map.keys())), [label_map[i] for i in label_map.keys()], rotation=90)
    plt.ylabel('Hidden States')
    plt.title('Emission Probability Matrix')
    plt.show()

    return optimal_model, clusters