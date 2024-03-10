import pandas as pd
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from similarity import are_similar
from sklearn.cluster import KMeans
import numpy as np

k = 20 #number of top ingredients to display
thresh = 0.8 #cosine similarity threshold
_SIMILARITY = True

def standardize_name(name):
    """Standardize the ingredient name for consistent comparison."""
    return name.lower().replace(" ", "")

def find_similar_node(G, name):
    """Find a node in the graph that is similar to the given name, or add a new node."""
    standardized_name = standardize_name(name)
    for node in G.nodes:
        if _SIMILARITY and are_similar(standardized_name, standardize_name(node), thresh):
            return node
    G.add_node(name)  # Add the node if no similar node is found
    return name

def preprocess_ingredients(ing_list):
    """
    Preprocess the ingredient list to group similar ingredients.
    Returns a dictionary with grouped ingredients.
    """
    grouped_ingredients = {}
    for ing in ing_list:
        standardized_ing = standardize_name(ing)
        found = False
        for key in grouped_ingredients:
            if are_similar(standardized_ing, key, thresh):
                grouped_ingredients[key].append(ing)
                found = True
                break
        if not found:
            grouped_ingredients[standardized_ing] = [ing]
    return grouped_ingredients

def kemeny_constant(G):
    path_length_dict = dict(nx.all_pairs_shortest_path_length(G))
    total_path_length = sum(sum(lengths.values()) for lengths in path_length_dict.values())
    num_pairs = len(G) * (len(G) - 1)
    return total_path_length / num_pairs

def coreness_based_on_kemeny(G):
    original_kemeny = kemeny_constant(G)
    coreness_scores = {}

    for node in G.nodes():
        H = G.copy()
        H.remove_node(node)
        new_kemeny = kemeny_constant(H)
        coreness_scores[node] = original_kemeny - new_kemeny

    return coreness_scores

def graph_update(G, ing_list):
    grouped_ings = preprocess_ingredients([x[0] for x in ing_list])
    for standard_ing, ings in grouped_ings.items():
        G.add_node(standard_ing)  # Add standardized node
        for ing1, ing2 in combinations(ings, 2):
            if G.has_edge(ing1, ing2):
                G[ing1][ing2]['weight'] += 1
            else:
                G.add_edge(ing1, ing2, weight=1)

def bipartite(file):

    #Read CSV to DataFrame
    df = pd.read_csv(file)
    assert 'Processed Ingredients' in df.columns, "DataFrame must contain a 'Processed Ingredients' column"

    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    #Initialize graph object
    G = nx.Graph()

    #Pull all processed ingredients into a graph structure
    for index,row in shuffled_df.iterrows():
        print(index)
        try:
            assert type(eval(row['Processed Ingredients'])) == list
            ing_list = eval(row['Processed Ingredients'])
            graph_update(G, ing_list)

        except Exception as e:
            print(index, " | ", row['ID'])
            print(e)

    # Top Nodes based on Coreness
    d = dict(G.degree)
    top_nodes = sorted(G.nodes, key=lambda x: d[x], reverse=True)[:k]
    print(top_nodes)
    input()

    # Coreness Metric Calculation
    degree_centrality = nx.degree_centrality(G)
    coreness_scores = coreness_based_on_kemeny(G)
    coreness_array = np.array(list(coreness_scores.values())).reshape(-1, 1)

    # Perform KMeans Clustering
    kmeans = KMeans(n_clusters=2, random_state=0).fit(coreness_array) #input is not appropriate
    labels = kmeans.labels_

    # Identify Core Ingredients
    # Assuming the cluster with the higher mean centrality score is the core cluster
    core_cluster = np.argmax(kmeans.cluster_centers_)
    core_ingredients = [node for node, label in zip(degree_centrality.keys(), labels) if label == core_cluster]

    # Top Nodes based on Coreness
    top_nodes = sorted(core_ingredients, key=lambda x: degree_centrality[x], reverse=True)[:k]

    # Create a subgraph with top k nodes
    H = G.subgraph(top_nodes)

    for i in top_nodes:
        print(i)

    # Drawing the subgraph
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(H)  # positions for all nodes 

    # Nodes
    nx.draw_networkx_nodes(H, pos, node_color='blue')

    # Edges
    nx.draw_networkx_edges(H, pos, width=1.0, alpha=0.5, edge_color='black')

    # Labels
    nx.draw_networkx_labels(H, pos, font_size=20, font_family='sans-serif')

    plt.axis('off')
    plt.show()
