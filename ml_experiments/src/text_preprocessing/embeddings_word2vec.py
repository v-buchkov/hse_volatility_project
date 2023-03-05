import numpy as np
import pandas as pd
from gensim.models import word2vec, Phrases
from gensim.models.phrases import Phraser

from ml_experiments.src.text_preprocessing.lemmatizer import add_lemmas_
from ml_experiments.src.text_preprocessing.pretrained import get_embedding_for_pretrained


def w2v_embedding(texts_df: pd.DataFrame, embedding_size: int = 200) -> np.array:
    add_lemmas_(texts_df)
    tokenized = [s.split() for s in texts_df['lemmas'].values]

    bigram = Phrases(tokenized, min_count=1, threshold=2)
    bigram_phraser = Phraser(bigram)
    tokenized += [bigram_phraser[t] for t in tokenized]

    w2v = word2vec.Word2Vec(tokenized, workers=4, vector_size=embedding_size, min_count=10, window=3, sample=1e-3)
    texts_df['embedding'] = texts_df['lemmas'].apply(lambda x: get_embedding_for_pretrained(x, model=w2v.wv, embedding_size=embedding_size))
    return np.array(list(texts_df['embedding'].values))
