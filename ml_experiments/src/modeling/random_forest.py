from typing import Tuple

from sklearn.ensemble import RandomForestClassifier


def random_forest(X_train, y_train, X_val, y_val, X_test, y_test, max_features: int = 4, n_estimators: int = 50,
                  random_state: int = 12) -> Tuple[float, float, float]:
    rf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    train_score = (y_train == rf.predict(X_train)).mean()

    val_score = (y_val == rf.predict(X_val)).mean()
    test_score = (y_test == rf.predict(X_test)).mean()

    return train_score, val_score, test_score
