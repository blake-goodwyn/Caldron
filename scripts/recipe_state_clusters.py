import pandas as pd
import numpy as np
import os
import pickle
import logging
import warnings
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import asyncio
from class_defs import RecipeAction, StateCluster

from genai_tools import limited_call, actions_extraction, cluster_label

# Initialize spaCy model once and use it throughout
method = "kmeans"  # Clustering method to use
RAW_TEXT_WEIGHT = 0.1  # Weight of the raw text embedding in the final action embedding

def process_instructions(instructions, retry_limit=3):
    print("Extracting actions from recipe instructions...")
    for _ in range(retry_limit):
        try:
            action_list = eval(instructions)
            return [action.lower() for action in action_list if isinstance(action, str)]
        except Exception as e:
            logging.warning(f"Error in action extraction: {e}. Retrying...")
    raise ValueError("Max retries reached for action extraction.")

def perform_clustering(recipe_actions, n_clusters_range):
    print(f"Clustering {len(recipe_actions)} recipe actions...")
    
    ## !!IMPORTANT!! ##
    embeddings = [np.concatenate([[action.pos], action.label_embedding, action.text_embedding * RAW_TEXT_WEIGHT]) for action in recipe_actions]
    # This is the main data structure that will be used for clustering.

    normalized_embeddings = normalize(embeddings)
    silhouette_avg_scores = []

    for n_clusters in n_clusters_range:
        print(f"Refining clusters with {n_clusters} clusters using {method}...")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)
                
                if method == 'kmeans':
                    clustering_model = KMeans(n_clusters=n_clusters, random_state=10)
                elif method == 'gmm':
                    clustering_model = GaussianMixture(n_components=n_clusters, random_state=10)
                else:
                    raise ValueError("Invalid clustering method selected.")

                clustering_model.fit(normalized_embeddings)
                if method == 'kmeans':
                    cluster_labels = clustering_model.labels_
                    clusters = [StateCluster(i, f"Cluster_{i}", clustering_model.cluster_centers_[i]) for i in range(n_clusters)]
                elif method == 'gmm':
                    cluster_labels = clustering_model.predict(normalized_embeddings)
                    clusters = [StateCluster(i, f"Cluster_{i}", clustering_model.means_[i]) for i in range(n_clusters)]

            silhouette_avg = silhouette_score(normalized_embeddings, cluster_labels)
            silhouette_avg_scores.append((n_clusters, silhouette_avg))

        except ConvergenceWarning:
            print(f"ConvergenceWarning thrown at {n_clusters} clusters. Stopping refinement.")
            break
    
    # Return cluster scores and list corresponding to best fit
    print(f"Returning silhouette scores and clusters...")
    best_fit = max(silhouette_avg_scores, key=lambda x: x[1])  # (n_clusters, silhouette_avg)
    if method == 'kmeans':
        clustering_model = KMeans(n_clusters=best_fit[0], random_state=10)
    elif method == 'gmm':
        clustering_model = GaussianMixture(n_components=best_fit[0], random_state=10)
    else:
        raise ValueError("Invalid clustering method selected.")

    clustering_model.fit(normalized_embeddings)
    if method == 'kmeans':
        cluster_labels = clustering_model.labels_
        clusters = [StateCluster(i, f"Cluster_{i}", clustering_model.cluster_centers_[i]) for i in range(best_fit[0])]
    elif method == 'gmm':
        cluster_labels = clustering_model.predict(normalized_embeddings)
        clusters = [StateCluster(i, f"Cluster_{i}", clustering_model.means_[i]) for i in range(best_fit[0])]

    # Clear actions in each cluster before assigning new actions
    for cluster in clusters:
        cluster.actions.clear()
    for action, label in zip(recipe_actions, cluster_labels):
        clusters[label].add_action(action)
        action.set_state(clusters[label].clusterID)

    return silhouette_avg_scores, clusters

async def create_actions(count, l, recipe):
    c = 0
    actions = []
    print(f"Processing recipe {recipe['ID']}")
    
    instructions = await limited_call(process_instructions, recipe.get('Instructions', ''))
    
    for i in instructions:
        extracted_actions = await limited_call(actions_extraction, i)
        for a in extracted_actions:
            c += 1
            #print(f"Processing action {c} of recipe {recipe['ID']} | #{index} of {l}")
            r = RecipeAction(recipe['ID'], c, i, a)
            actions.append(r)

    # Normalize the positions of the actions depending on the length of recipe_actions list
    for r in actions:
        r.normalize_position(len(actions))

    count['done'] += 1
    print(f"Completed recipe #{count['done']} of {l}")
    
    return actions

async def process_recipes(file_path, k, c):
    print(f"Processing {k} recipes from {file_path}...")
    df = pd.read_csv(file_path)
    if k > len(df):
        k = len(df)
    recipes = df.sample(frac=1).head(k).to_dict(orient='records')

    recipe_actions = await asyncio.gather(*(create_actions(c, len(recipes), recipe) for index, recipe in enumerate(recipes)))
    
    return recipe_actions

async def label_cluster(cluster):
    
    action_prompt = ""
    for action in cluster.actions:
        temp = str(action.pos) + " | "
        temp += str(action.label) + "\n"
        action_prompt += temp
    
    new_label = await limited_call(cluster_label, action_prompt)
    cluster.set_label(new_label.lower().strip().split()[0])

async def process_clusters(clusters):
    return await asyncio.gather(*(label_cluster(cluster) for cluster in clusters))

def find_state_clusters(file, sample=2000, max_clusters=150):
    print("Starting recipe modeling...")
    counter = {"done": 0}
    try:
        recipe_actions = asyncio.run(process_recipes(file, sample, counter))
        all_actions = [item for sub in recipe_actions for item in sub]
        print(f"{len(all_actions)} recipe actions extracted.")
        n_clusters_range = range(2, max_clusters, 2)
        scores, clusters = perform_clustering(all_actions, n_clusters_range)
        print("Best Fit Score: ", max(scores, key=lambda x: x[1]))

        ##Make sense of the cluster labels with a single label
        asyncio.run(process_clusters(clusters))

        # Define the output file path
        cwd = os.getcwd()
        output_directory = os.path.join(cwd, 'outputs')
        
        #pickle the clusters and recipe_actions
        with open(os.path.join(output_directory, 'clusters.pkl'), 'wb') as file:
            pickle.dump(clusters, file)
        
        with open(os.path.join(output_directory, 'recipe_actions.pkl'), 'wb') as file:
            pickle.dump(recipe_actions, file)

        output_file_path = os.path.join(output_directory, 'recipe_sequences.txt')

        # Check if the directory exists, create it if it doesn't
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        with open(output_file_path, 'w+') as file:
            for action_list in recipe_actions:
                file.write("<")
                print("<", end="")
                for action in action_list:
                    file.write(f"{action.state} ")
                    print(f"{action.state} ", end="")
                file.write(">\n")  # Write a newline character after each action list
                print(">")

        ## Plotting and additional processing can go here ##
        plt.plot(*zip(*scores))
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score vs Number of Clusters")
        plt.show()

    except Exception as e:
        logging.error(f"Error in processing: {e}")