import pandas as pd
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from similarity import are_similar, sim_embedding, shared_embeddings
from dataset_linter import safely_convert_to_list
import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import asyncio
import pickle
import os
from time import sleep

k = 20 #number of top ingredients to display
thresh = 0.8 #cosine similarity threshold
_SIMILARITY = True

def standardize_name(name):
    """Standardize the ingredient name for consistent comparison."""
    return name.lower()

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
                print("GROUPED! ", ing, " | ", key)
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
    grouped_ings = [x[0] for x in ing_list]
    for ing in grouped_ings:
        for ing1, ing2 in combinations(grouped_ings, 2):
            if G.has_edge(ing1, ing2):
                G[ing1][ing2]['weight'] += 1
            else:
                G.add_edge(ing1, ing2, weight=1)

def combine_similar_nodes(graph):

    nodes_to_combine = []
    removed_nodes = set()
    all_nodes = list(graph.nodes)
    
    # Wrap the outer loop with tqdm to show a progress bar
    for i, node1 in tqdm(enumerate(all_nodes[:-1]), total=len(all_nodes)-1, desc='Combining nodes'):
        if node1 in removed_nodes:
            continue
        for node2 in all_nodes[i+1:]:
            if node2 in removed_nodes or not are_similar(node1, node2):
                continue
            keep_node = min(node1, node2, key=len)
            remove_node = node2 if keep_node == node1 else node1
            nodes_to_combine.append((keep_node, remove_node))
            removed_nodes.add(remove_node)
    
    # Combine nodes based on the gathered pairs
    for keep_node, remove_node in tqdm(nodes_to_combine, desc='Merging edges'):
        if remove_node not in graph:
            continue
        
        for neighbor, data in list(graph[remove_node].items()):
            if neighbor != keep_node:
                if graph.has_edge(keep_node, neighbor):
                    graph[keep_node][neighbor].update(data)
                else:
                    graph.add_edge(keep_node, neighbor, **data)
        graph.remove_node(remove_node)
    
    return graph

async def form_sim_embeddings(G):
    
    for node in tqdm(G.nodes, desc='Creating embeddings'):
        node_tasks = await asyncio.create_task(asyncio.to_thread(sim_embedding,node))

    tqdm_asyncio.as_completed(asyncio.wait(node_tasks))

def bipartite(file, opt="FOOD_DATA"):

    #Read CSV to DataFrame
    df = pd.read_csv(file)
    assert 'Processed Ingredients' in df.columns, "DataFrame must contain a 'Processed Ingredients' column"

    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    #Initialize graph object
    G = nx.Graph()

    #Pull all processed ingredients into a graph structure
    for index,row in shuffled_df.iterrows():
        print(index, end='')
        ing_list = []
        try:
            assert type(eval(row['Processed Ingredients'])) == list
            for i in eval(row['Processed Ingredients']):
                assert type(i[0]) == str
            ing_list = eval(row['Processed Ingredients'])
        except (AssertionError, Exception) as e:
            ing_list = safely_convert_to_list(row['Processed Ingredients'])
            try:
                assert type(ing_list) == list, "Processed Ingredients must be a list"
                for i in ing_list:
                    assert type(i[0]) == str
            except (AssertionError, Exception) as e:
                #print(f" Error: {e}")
                ing_list = []

        if ing_list != []:
            
            graph_update(G, ing_list)

    #Create embeddings for each ingredient
    asyncio.run(form_sim_embeddings(G))
    print(f"Embeddings created: {len(shared_embeddings.keys())}")
    G = combine_similar_nodes(G)

    cwd = os.getcwd()

    if opt == "FOOD_DATA":
        
        #get all existing terms in the file
        with open(os.path.join(cwd, 'ingredients', 'search_terms.txt'), 'r') as file:
            search_terms = file.readlines()

        #convert search terms to a set
        search_terms_set = set([term.strip() for term in search_terms])

        #add all nodes from graph into the set
        for node in G.nodes:
            try:
                search_terms_set.add(node)
            except Exception as e:
                print(f"Error: {e}")

        print(f"Search terms: {len(search_terms_set)}")

        sleep(2)

        #write all terms to a file
        output_directory = os.path.join(cwd, 'ingredients')
        with open(os.path.join(output_directory, 'search_terms.txt'), 'w') as file:
            for term in search_terms_set:
                try:
                    file.write(term + '\n')
                except Exception as e:
                    print(f"Error: {e}")
        
        return

    elif opt == "PLOT":
        # Define the output file path
        output_directory = os.path.join(cwd, 'outputs')

        #pickle the clusters and recipe_actions
        with open(os.path.join(output_directory, 'ingredient_bipartite.pkl'), 'wb') as file:
            pickle.dump(G, file)
        
        # Top Nodes based on Coreness
        degrees = dict(G.degree)
        threshold = np.quantile(list(degrees.values()), 0.99)
        core_ingredients = sorted([node for node, degree in degrees.items() if degree >= threshold], key=lambda x: degrees[x], reverse=True)

        # Create a subgraph with top k nodes
        H = G.subgraph(core_ingredients)

        for i in sorted(H.nodes, key=lambda x: degrees[x], reverse=True):
            print(f"{i} | {degrees[i]}")
        
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
    else:
        print("Invalid option")
        return
