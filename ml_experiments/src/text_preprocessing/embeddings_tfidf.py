from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ml_experiments.src.text_preprocessing.lemmatizer import add_lemmas_


def tfidf_embedding(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                    ngram_range: Tuple[int, int] = (1, 2)) -> Tuple[np.array, np.array, np.array]:
    for text in (X_train, X_val, X_test):
        add_lemmas_(text)

    tf_idf = TfidfVectorizer(ngram_range=ngram_range)
    X_train = tf_idf.fit_transform(X_train['lemmas'])
    X_val = tf_idf.transform(X_val['lemmas'])
    X_test = tf_idf.transform(X_test['lemmas'])
    return X_train, X_val, X_test
