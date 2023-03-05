import os

import numpy as np
import pandas as pd
import fasttext
import fasttext.util
from tqdm.auto import tqdm

from ml_experiments.src.text_preprocessing.lemmatizer import add_lemmas_
from ml_experiments.src.text_preprocessing.pretrained import get_embedding_for_pretrained

import warnings
warnings.filterwarnings('ignore')


def fasttext_embedding(texts_df: pd.DataFrame, language: str = 'ru', model_name: str = 'cc.ru.300.bin',
                       embedding_size=300) -> np.array:
    if model_name not in os.listdir():
        fasttext.util.download_model(language, if_exists='ignore')
    ft = fasttext.load_model(model_name)

    add_lemmas_(texts_df)
    tqdm.pandas()
    texts_df['embedding'] = texts_df['lemmas'].apply(lambda x: get_embedding_for_pretrained(x, model=ft, embedding_size=embedding_size))
    return np.array(list(texts_df['embedding'].values))
