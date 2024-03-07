import pandas as pd
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
import math
import numpy as np
from collections import Counter
import spacy

nlp = spacy.load('en_core_web_md')
cosThreshold = 0.8 #arbitrary threshold for cosine similarity
k = 20 #top ingredients

def graph_update(G, ing_list):
    
    l = [x[0] for x in ing_list]

    for ing1, ing2 in combinations(l, 2):

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
        for existing_ing in C:
            if are_similar(existing_ing, ing): #checks cosine similarity
                #print("similar! : ", existing_ing, " | ", ing)
                C[existing_ing] += 1
                found = True
                break
        if not found:
            C[ing] += 1

## Quick Cosine Similarity
def are_similar(str1, str2):

    #get embeddings for the two strings
    u = nlp(str1.lower()).vector
    v = nlp(str2.lower()).vector

    assert(u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu!=0 and vv!=0:
        cos_theta = uv/np.sqrt(uu*vv)

    return cos_theta > cosThreshold

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

        if index >= 50:
            break

    #for i in sorted(C, key=lambda x: C[x], reverse=True)[:k]:
    #    print(i, " : ", C[i])
    # 
    #input("Press Enter to continue...")

    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:k]
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