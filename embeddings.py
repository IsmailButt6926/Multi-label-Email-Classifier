# Methods related to converting text into numeric representation and then returning numeric representation
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Config import Config


def get_tfidf_embd(df):
    """Convert text data to TF-IDF embeddings.

    Combines Ticket Summary and Interaction content into a single corpus,
    then applies TF-IDF vectorization.

    Returns:
        np.ndarray: Dense TF-IDF feature matrix.
    """
    # Combine text fields into a single corpus
    corpus = (df[Config.TICKET_SUMMARY].fillna('') + ' ' +
              df[Config.INTERACTION_CONTENT].fillna(''))

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(corpus)

    return X.toarray()
