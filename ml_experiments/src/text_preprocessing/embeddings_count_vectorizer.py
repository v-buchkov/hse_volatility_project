from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from ml_experiments.src.text_preprocessing.lemmatizer import add_lemmas_


def count_vectorizer_embedding(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                               ngram_range: Tuple[int, int] = (1, 2)) -> Tuple[np.array, np.array, np.array]:
    for text in (X_train, X_val, X_test):
        add_lemmas_(text)

    count_vectorizer = CountVectorizer(ngram_range=ngram_range)
    X_train = count_vectorizer.fit_transform(X_train['lemmas'])
    X_val = count_vectorizer.transform(X_val['lemmas'])
    X_test = count_vectorizer.transform(X_test['lemmas'])
    return X_train, X_val, X_test
