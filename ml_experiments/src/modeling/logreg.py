from typing import Tuple, Union, Any

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def logreg(X_train, y_train, X_val, y_val, X_test, y_test,
           quality_metric: Union[Any, str] = 'default', random_state: int = 12) -> Tuple[float, float, float]:
    if quality_metric == 'default':
        quality_metric = accuracy_score

    clf = LogisticRegression(random_state=random_state)
    clf.fit(X_train, y_train)
    train_score = quality_metric(clf.predict(X_train), y_train)

    val_score = quality_metric(clf.predict(X_val), y_val)
    test_score = quality_metric(clf.predict(X_test), y_test)

    return train_score, val_score, test_score
