import pandas as pd
import spacy
import numpy as np
import os
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from genai_tools import actions_extraction, extract_action, cluster_label
import logging
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
import tiktoken
import pickle

# Initialize spaCy model once and use it throughout
nlp = spacy.load('en_core_web_md')
method = "kmeans"  # Clustering method to use
tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')

class RecipeAction:
    def __init__(self, ID, position, text, label):
        #print(f"Creating action for recipe {ID} at position {position} with text '{text}'...")
        self.recipeID = ID      # ID of the recipe
        self.pos = position     # Position in the recipe instructions    
        self.text = text        # Text of the recipe instruction
        self.label = label          # One-word action label of the instruction
        self.label_embedding = self.generate_embedding(label)  # Embedding of the action label
        self.text_embedding = self.generate_embedding(text)  # Embedding of the instruction text
        self.state = None       # Defined state of the action from cluster
        #print(self)

    def generate_embedding(self, text):
        """Generate and return the embedding for the given text."""
        return nlp(text).vector
    
    def set_label(self, label):
        """Generate and return the action label for the given text."""
        self.label = label

    def normalize_position(self, length):
        """Normalize the position of the action based on the length of the recipe."""
        self.pos = self.pos / length

    def embedding(self):
        """Return the embedding of the recipe step."""
        return self.embedding

    def set_state(self, state):
        """Set the state of the recipe step."""
        print(f"Setting state of action {self.label} to {state}")
        self.state = state

    def update_text(self, new_text, new_label):
        """Update the text of the recipe step and regenerate its embedding."""
        self.text = new_text
        self.embedding = self.generate_embedding(new_text)
        self.label = new_label

    def __str__(self):
        """String representation of the recipe step."""
        return f"Recipe ID: {self.recipeID} | Position: {self.pos}, Label: {self.label}"

    def __eq__(self, other):
        """Equality check based on recipe ID and position."""
        return self.recipeID == other.recipeID and self.pos == other.pos

    def __lt__(self, other):
        """Less than comparison based on position in the recipe."""
        return self.pos < other.pos

class StateCluster:
    def __init__(self, ID, label, centroid):
        #print(f"Creating cluster {ID} with label '{label}'")
        self.clusterID = ID         # ID of the cluster
        self.position = None         # Position derived from the average position of actions in the cluster
        self.label = label          # Label of the cluster
        self.centroid = centroid    # Centroid of the cluster
        self.actions = []      # List of RecipeAction objects in the cluster

    def add_action(self, action):
        """Add an action to the cluster."""
        print(f"Adding action {action.label} to cluster {self.clusterID}")
        assert type(action) == RecipeAction, "Input must be a RecipeAction object"
        self.actions.append(action)
        self.position = np.mean([action.pos for action in self.actions])

    def remove_action(self, action):
        """Remove an action from the cluster."""
        self.actions.remove(action)

    def position(self):
        """Return the position of the cluster."""
        return self.position.__format__(".2f")

    def actions(self):
        """Return the list of actions in the cluster."""
        return self.actions
    
    def set_label(self, label):
        """Set the label of the cluster."""
        self.label = label

    def update_centroid(self):
        if self.actions:
            self.centroid = np.mean([action.embedding for action in self.actions], axis=0)
        else:
            self.centroid = np.zeros_like(self.centroid)

    def __str__(self):
        """String representation of the cluster."""
        return f"Cluster ID: {self.clusterID} | Label: {self.label}, Actions: {self.actions}"

#####

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
    #print(recipe_actions[0].pos, recipe_actions[0].label_embedding, recipe_actions[0].text_embedding)
    embeddings = [np.concatenate([[action.pos], action.label_embedding]) for action in recipe_actions]
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

def process_recipes(file_path, k):
    print(f"Processing {k} recipes from {file_path}...")
    df = pd.read_csv(file_path)
    recipes = df.sample(frac=1).head(k).to_dict(orient='records')

    recipe_actions = []
    for recipe in recipes: 
        c = 0
        new = []
        for i in process_instructions(recipe.get('Instructions', '')):
            for a in actions_extraction(i):
                c += 1
                print(f"Processing action {c} of recipe {recipe['ID']}...")
                r = RecipeAction(recipe['ID'], c, i, a)
                new.append(r)
                #normalize the positions of the actions depending on the length of recipe_actions list
        for r in new:
            r.normalize_position(len(new))
        recipe_actions.append(new)       

    return recipe_actions

def main(file, k=1000):
    print("Starting recipe modeling...")
    
    try:
        recipe_actions = process_recipes(file, k)
        all_actions = [item for sub in recipe_actions for item in sub]
        print(f"{len(all_actions)} recipe actions extracted.")
        n_clusters_range = range(2, 500)
        scores, clusters = perform_clustering(all_actions, n_clusters_range)
        print("Best Fit Score: ", max(scores, key=lambda x: x[1]))

        # Define the output file path
        cwd = os.getcwd()
        output_directory = os.path.join(cwd, 'outputs')

        #pickle the clusters and recipe_actions
        with open(os.path.join(output_directory, 'clusters.pkl'), 'wb') as file:
            pickle.dump(clusters, file)
        
        with open(os.path.join(output_directory, 'recipe_actions.pkl'), 'wb') as file:
            pickle.dump(recipe_actions, file)

        ##Make sense of the cluster labels with a single label
        for cluster in sorted(clusters, key=lambda x: x.position):
            #for each of the clusters, find appropriate label
            cluster_actions = cluster.actions #form action list
            action_prompt = ""
            for action in cluster_actions:
                temp = str(action.pos) + " | "
                temp += str(action.label) + "\n"
                if len(tokenizer.encode(action_prompt+temp)) < 16385: #openapi token limit
                    action_prompt += temp
                else:
                    break

            #ensure that action_prompt is less than 16385 tokens
            cluster.set_label(cluster_label(action_prompt).lower())
            print(f"Cluster ID: {cluster.clusterID} | Label: {cluster.label} | Position: {cluster.position}| # of Actions: {len(cluster.actions)}")

        output_file_path = os.path.join(output_directory, 'recipe_actions.txt')

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
        

        # Plotting and additional processing can go here
        plt.plot(*zip(*scores))
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score vs Number of Clusters")
        plt.show()

    except Exception as e:
        logging.error(f"Error in processing: {e}")

if __name__ == "__main__":
    main('data/banana-bread/processed-banana-bread-recipes-2024-02-27-1547.csv')