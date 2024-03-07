import pandas as pd
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from cos_similarity import are_similar

k = 20 #top ingredients
thresh = 0.8 #cosine similarity threshold
_SIMILARITY = False

def graph_update(G, ing_list):
    
    l = [x[0] for x in ing_list]

    for ing1, ing2 in combinations(l, 2):

        if _SIMILARITY:
            if are_similar(ing1, ing2):
                print("similar! : ", ing1, " | ", ing2)
        if G.has_edge(ing1, ing2): # If the edge exists, increase weight; otherwise, add edge with weight 1
            G[ing1][ing2]['weight'] += 1
        else:
            G.add_edge(ing1, ing2, weight=1)

def counter_update(C, ing_list):

    l = [x[0] for x in ing_list] #grabs just the ingredients
    for ing in l: #iterate across list of ingredients
        found = False
        if _SIMILARITY:
            for existing_ing in C:
                if are_similar(existing_ing, ing): #checks cosine similarity
                    C[existing_ing] += 1
                    found = True
                    break
        if not found:
            C[ing] += 1

def bipartite(file):

    #Read CSV to DataFrame
    df = pd.read_csv(file)
    assert 'Processed Ingredients' in df.columns, "DataFrame must contain a 'Processed Ingredients' column"

    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    #Initialize graph object
    C = Counter()
    G = nx.Graph()

    #Pull all processed ingredients into a graph structure
    for index,row in shuffled_df.iterrows():
        print(index)
        try:
            assert type(eval(row['Processed Ingredients'])) == list
            ing_list = eval(row['Processed Ingredients'])
            graph_update(G, ing_list)
            counter_update(C, ing_list)

        except Exception as e:
            print(index, " | ", row['ID'])
            print(e)

    degrees = dict(G.degree())
    nodes = sorted(degrees, key=degrees.get, reverse=True)
    top_nodes = []

    while len(top_nodes) < k and nodes:
        current_node = nodes.pop(0)
        merged = False

        for i in range(len(top_nodes)):
            if are_similar(current_node, top_nodes[i], cosThreshold=thresh):
                # Merge nodes by summing their degrees
                degrees[top_nodes[i]] += degrees[current_node]
                merged = True
                break

        if not merged:
            top_nodes.append(current_node)

    # Sort top_nodes by their updated degrees if required
    top_nodes = sorted(top_nodes, key=lambda n: degrees[n], reverse=True)
                
    print("Top nodes: ", top_nodes)
    input()

    for ingredient, count in C.items():
        if ingredient in G:
            G.nodes[ingredient]['size'] = count

    # Create a subgraph with top 50 nodes
    H = G.subgraph(top_nodes)

    for i in top_nodes:
        print(i)

    # Drawing the subgraph
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(H)  # positions for all nodes 

    # Nodes
    nx.draw_networkx_nodes(H, pos, node_size=[H.nodes[n]['size'] for n in H], node_color='blue')

    # Edges
    nx.draw_networkx_edges(H, pos, width=1.0, alpha=0.5, edge_color='black')

    # Labels
    nx.draw_networkx_labels(H, pos, font_size=20, font_family='sans-serif')

    plt.axis('off')
    plt.show()

bipartite('data/GOOD DATASETS/processed-banana_bread-recipes-2024-02-27-1547.csv')