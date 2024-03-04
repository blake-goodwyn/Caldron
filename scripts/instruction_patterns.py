#Converts given instructions represented as strings into embeddings. Then finds patterns in the embeddings and returns a list of the patterns found.

import pandas as pd
from genai_tools import get_embedding_async
import asyncio
import numpy as np
import hmmlearn
import time

# Reads strings from a file and returns a list of the strings.
def read_strings_from_file(file_path):
    """
    Reads strings from a file and returns a list of the strings.
    
    Args:
        file_path (str): The path to the file to read from.
    
    Returns:
        list: A list of strings read from the file.
    """
    df = pd.read_csv(file_path)
    return df['Instructions'].tolist()

async def embeddings_list(l):
    return await asyncio.gather(*[get_embedding_async(s) for s in l])

l = read_strings_from_file('data/tart/processed-tart-recipe-2024-02-29-2229.csv')
new_list = []
for i in l:
    t = eval(i)
    temp = []
    time1 = time.time()
    temp = asyncio.run(embeddings_list(t))
    time2 = time.time()
    print(time2-time1)
    new_list.append(temp)

for i in new_list:
    print(i)
    
# Converts a list of strings into embeddings
def convert_strings_to_embeddings(strings):
    """
    Converts a list of strings into embeddings.
    
    Args:
        strings (list): A list of strings to convert into embeddings.
    
    Returns:
        list: A list of embeddings for the input strings.
    """
    return [get_embedding(s) for s in strings]

# Finds patterns in the embeddings via Hidden Markov Models and returns a list of the patterns found
def find_patterns_in_embeddings(embeddings):
    """
    Finds patterns in the embeddings via Hidden Markov Models and returns a list of the patterns found.
    
    Args:
        embeddings (list): A list of embeddings to find patterns in.
    
    Returns:
        list: A list of patterns found in the embeddings.
    """
    return []