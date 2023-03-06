from typing import Tuple
import re

import numpy as np
import pandas as pd
from gensim.models import Phrases
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phraser

from ml_experiments.src.text_preprocessing.lemmatizer import add_lemmas_


class Word2VecTransformer:

    def __init__(self, w2v_model, word_pattern):

        self.w2v_model = w2v_model
        self.word_pattern = word_pattern
        self.re = re.compile(pattern=self.word_pattern)

    def fit(self, X):
        return self

    def transform(self, X):
        X_transformed = np.zeros((len(X), self.w2v_model.wv.vector_size))
        for i, title in enumerate(X):
            title_vector = np.zeros((self.w2v_model.wv.vector_size,))
            tokens = self.re.findall(title.lower())
            for token in tokens:
                if token in self.w2v_model.wv.key_to_index:
                    title_vector += self.w2v_model.wv.get_vector(token)

            X_transformed[i] = title_vector

        return X_transformed


def w2v_embedding(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                  n_epochs: int = 20) -> Tuple[np.array, np.array, np.array]:
    WORD_PATTERN = '(?u)\\b\\w\\w+\\b'

    for text in (X_train, X_val, X_test):
        add_lemmas_(text)
        text['tokenized'] = [s.split() for s in text['lemmas'].values]

    tokenized = X_train['tokenized']
    bigram = Phrases(tokenized, min_count=1, threshold=2)
    bigram_phraser = Phraser(bigram)
    tokenized += [bigram_phraser[t] for t in tokenized]

    w2v_model = Word2Vec(sg=1, )
    w2v_model.build_vocab(tokenized)
    w2v_model.train(
        X_train,
        total_examples=w2v_model.corpus_count,
        epochs=n_epochs,
        compute_loss=True
    )

    w2v_transformer = Word2VecTransformer(w2v_model=w2v_model, word_pattern=WORD_PATTERN)

    X_train = w2v_transformer.transform(X_train['lemmas'])
    X_val = w2v_transformer.transform(X_val['lemmas'])
    X_test = w2v_transformer.transform(X_test['lemmas'])

    return X_train, X_val, X_test
