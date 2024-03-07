#Converts given instructions represented as strings into a list of instruction states and resulting MArkov model for the given instructions

import pandas as pd
import asyncio
from genai_tools import action_extraction_async, action_extraction, get_embedding_async, get_embedding
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import spacy

nlp = spacy.load('en_core_web_md')

k = 50 #batch size
_PROCESS = "SPACY" #"OPENAI"

##Synchronous     
def process_instructions(recipe):
    temp = []
    while True:
        temp = action_extraction(recipe['instructions'])
        try:
            assert type(eval(temp)) == list
            break
        except Exception as e:
            print(e)
            print("Retrying action extraction...")  

    #print(temp)
    recipe['instr_list'] = temp
    return

def process_actions(recipe):
    out = []
    try:
        assert type(eval(recipe['instr_list'])) == list
        for a in eval(recipe['instr_list']):
            try:
                if _PROCESS == "OPENAI":
                    out.append(get_embedding(a))
                elif _PROCESS == "SPACY":
                    out.append(nlp(a).vector)
            except Exception as e:
                print(e)
                out.append([])
        #print(out)
        recipe['embeddings'] = out
        return
    except Exception as e:
        print(e)
        recipe['embeddings'] = []
        return

##Asynchronous
async def process_instructions_async(recipe):
    print(f"Processing instruction lists for recipe {recipe['index']+1}")
    temp = []
    while True:
        temp = action_extraction(recipe['instructions'])
        try:
            assert type(eval(temp)) == list
            for i in eval(temp):
                assert type(i) == str
            break
        except Exception as e:
            print(e)
            print("Retrying action extraction...")  

    #print(temp)
    return temp

async def process_actions_async(recipe):
    print(f"Processing action embeddings for recipe {recipe['index']+1}")
    out = []
    while True:
        try:
            assert type(eval(recipe['instr_list'])) == list
            for a in eval(recipe['instr_list']):
                try:
                    out.append(get_embedding(a))
                except:
                    out.append([])
            break
        except Exception as e:
            print(e)
            
    return out

async def process_recipe(recipe):
    recipe['instr_list'] = await process_instructions_async(recipe)
    recipe['embeddings'] = await process_actions_async(recipe)

async def process_recipe_batch(recipes):
    tasks = []
    for r in recipes:
        tasks.append(asyncio.create_task(process_recipe(r)))
    return await asyncio.gather(*tasks)

def markov(file):

    # 1. Preprocessing
    print(f"Reading file {file}")
    df = pd.read_csv(file)
    assert 'Instructions' in df.columns
    assert 'ID' in df.columns
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    recipes = []
    for index, row in shuffled_df.iterrows():
        recipes.append({"index": index, "id": row['ID'], "instructions": row['Instructions'], "instr_list": [], "embeddings": []})
        if index >= k:
            break

    # 2. Action Extraction (via GPT-3.5)
    print(f"Processing recipes")

    ##attempt at asynchronous processing
    #recipes = asyncio.run(process_recipe_batch(recipes))

    for r in recipes:
        print(f"Processing instruction lists for recipe {r['index']+1}")
        process_instructions(r)
        print(f"Processing action embeddings for recipe {r['index']+1}")
        process_actions(r)

    # 3. Action Clustering (via embeddings)
        
    ##extract embeddings from recipes
    #actions = [a for r in recipes for a in r['instr_list']]
        
    input("Extracting embeddings from recipes")

    embeddings = [e for r in recipes for e in r['embeddings']]

    #for i in embeddings:
    #    print(len(i))

    input("Normalizing embeddings")

    normalized_embeddings = normalize(embeddings)

    #for i in normalized_embeddings:
    #    print(i)

    print(np.shape(normalized_embeddings))

    # Perform clustering on the cosine similarity matrix
    range_n_clusters = range(2, len(normalized_embeddings)/2)  # Example range, can be adjusted
    silhouette_avg_scores = []

    for n_clusters in range_n_clusters:
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = kmeans.fit_predict(normalized_embeddings)

        # Calculate the silhouette score
        silhouette_avg = silhouette_score(normalized_embeddings, cluster_labels)
        silhouette_avg_scores.append(silhouette_avg)

        print(f"Number of clusters: {n_clusters}, Silhouette score: {silhouette_avg}")

    plt.plot(range_n_clusters, silhouette_avg_scores)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs Number of Clusters")
    plt.show()

    # 4. Markov Model Generation
            
    ##iterate over recipe embeddings and states to generate transition matrix

    ##generate Markov model from transition matrix

    # 5. Model Evaluation
            
    ##evaluate model via perplexity and other metrics

markov('data/banana-bread/processed-banana-bread-recipes-2024-02-27-1547.csv')