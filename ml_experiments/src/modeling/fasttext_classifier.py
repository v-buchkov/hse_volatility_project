from typing import Tuple

import fasttext
from sklearn.metrics import accuracy_score

from ml_experiments.src.text_preprocessing.lemmatizer import add_lemmas_

import warnings
warnings.filterwarnings('ignore')


def fasttext_classifier(X_train, y_train, X_val, y_val,
                        X_test, y_test) -> Tuple[float, float, float]:
    add_lemmas_(X_train)
    add_lemmas_(X_test)
    add_lemmas_(X_val)

    with open('train_ft.txt', 'w') as f:
        for label, lemmas in list(zip(
                y_train, X_train['lemmas']
        )):
            f.write(f"__label__{int(label)} {lemmas}\n")

    classifier = fasttext.train_supervised('train_ft.txt')
    train_score = accuracy_score([int(label[0][-1]) for label in classifier.predict(list(X_train['lemmas']))[0]], y_train)

    val_score = accuracy_score([int(label[0][-1]) for label in classifier.predict(list(X_val['lemmas']))[0]], y_val)
    test_score = accuracy_score([int(label[0][-1]) for label in classifier.predict(list(X_test['lemmas']))[0]], y_test)

    return train_score, val_score, test_score
