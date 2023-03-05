from typing import Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def logreg(X_train, y_train, X_val, y_val, X_test, y_test, random_state: int = 12) -> Tuple[float, float, float]:
    clf = LogisticRegression(random_state=random_state)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)

    val_score = accuracy_score(clf.predict(X_val), y_val)
    test_score = accuracy_score(clf.predict(X_test), y_test)

    return train_score, val_score, test_score
