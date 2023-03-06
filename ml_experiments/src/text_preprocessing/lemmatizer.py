import re
from functools import lru_cache

import pandas as pd
from tqdm import tqdm
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords

m = MorphAnalyzer()


def _words_only(text):
    regex = re.compile("[а-яa-zёЁ]+")
    try:
        return regex.findall(text.lower())
    except:
        return []


@lru_cache(maxsize=128)
def _lemmatize_word(token, pymorphy=m):
    return pymorphy.parse(token)[0].normal_form


def _lemmatize_text(text):
    return [_lemmatize_word(w) for w in text]


def remove_stopwords(lemmas, language: str = 'russian'):
    sw = stopwords.words(language)
    return [w for w in lemmas if not w in sw and len(w) > 3]


def _clean_text(text):
    tokens = _words_only(text)
    lemmas = _lemmatize_text(tokens)

    return ' '.join(remove_stopwords(lemmas))


def add_lemmas_(texts_df: pd.DataFrame) -> None:
    texts_df['lemmas'] = list(tqdm(map(_clean_text, texts_df['text']), total=len(texts_df)))
