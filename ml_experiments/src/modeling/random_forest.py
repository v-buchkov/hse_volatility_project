from typing import Tuple, Union, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def random_forest(X_train, y_train, X_val, y_val, X_test, y_test, max_features: int = 4, n_estimators: int = 50,
                  quality_metric: Union[Any, str] = 'default', random_state: int = 12) -> Tuple[float, float, float]:
    if quality_metric == 'default':
        quality_metric = accuracy_score

    rf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    train_score = quality_metric(rf.predict(X_train), y_train)

    val_score = quality_metric(rf.predict(X_val), y_val)
    test_score = quality_metric(rf.predict(X_test), y_test)

    return train_score, val_score, test_score
