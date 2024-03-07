import spacy
import numpy as np

nlp = spacy.load('en_core_web_md')

## Quick Cosine Similarity
def are_similar(str1, str2, cosThreshold=0.8, verbose=False):

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

    if verbose:
        print(f"Cosine Similarity between {str1} and {str2} is {cos_theta}")

    return cos_theta > cosThreshold