from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def tfidf_similarity(cv,s,texts):
    s,sents = ' '.join(list(s)), [' '.join(list(text)) for text in texts]
    corpus = [s] + sents
    vectors = cv.fit_transform(corpus).toarray()
    scores = cosine_similarity(vectors[0].reshape(1,-1),vectors[1:])
    # print(scores)
    sorted_indices = scores.argsort()[0,::-1]
    return texts[sorted_indices[0]]


