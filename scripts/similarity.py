import spacy
import numpy as np
from difflib import SequenceMatcher

nlp = spacy.load('en_core_web_md')

def standardize_str(s):
    return s.lower().replace("-", " ").replace("  ", " ")

def split_terms(s):
    return s.split()

## Quick Cosine Similarity
def are_similar(str1, str2, cosThreshold=0.8, verbose=False):

    str1 = standardize_str(str1)
    str2 = standardize_str(str2)

    # 1. Cosine Similarity
    if cosine_similar(str1, str2, cosThreshold):
        if verbose:
            print(f"Cosine similarity match: '{str1}' and '{str2}'")
        return True

    # 2. String Similarity (e.g., Substring match, Levenshtein distance)
    if string_similarity(str1, str2):
        if verbose:
            print(f"String similarity match: '{str1}' and '{str2}'")
        return True
    
    return False

def string_similarity(str1, str2, threshold=0.4):
    # Simple substring check or other string similarity measures
    # Example: Using SequenceMatcher from difflib
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio() > threshold

def cosine_similar(str1, str2, cosThreshold):
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