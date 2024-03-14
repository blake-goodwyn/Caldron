import spacy
import numpy as np
from difflib import SequenceMatcher

nlp = spacy.load('en_core_web_md')
shared_embeddings = {}

def sim_embedding(ing):
    assert type(ing) == str, "Input must be a string"
    #shared_embeddings[ing] = get_embedding(ing)
    shared_embeddings[ing] = nlp(ing).vector

def standardize_str(s):
    return s.lower().replace("-", " ").replace("  ", " ")

def split_terms(s):
    return s.split()

## Quick Cosine Similarity
def are_similar(str1, str2, threshold=2):

    vec1 = shared_embeddings.get(str1, None)
    vec2 = shared_embeddings.get(str2, None)

    if vec1 is None or vec2 is None:
        return False

    # 1. Cosine Similarity
    if cosine_similar(vec1,vec2) + string_similarity(str1, str2) > threshold:
        return True
    
    return False

def string_similarity(str1, str2):
    # Simple substring check or other string similarity measures
    # Example: Using SequenceMatcher from difflib
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def cosine_similar(vec1, vec2):
    #get embeddings for the two strings
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)