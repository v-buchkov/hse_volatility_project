from typing import Tuple

import catboost as cb
from sklearn.metrics import accuracy_score


def catboosting(X_train, y_train, X_val, y_val, X_test, y_test, random_state: int = 12) -> Tuple[float, float, float]:
    params = dict(
        learning_rate=0.025,
        iterations=10000,
        reg_lambda=0.0005,
        colsample_bylevel=1.,
        max_bin=80,
        bagging_temperature=2,
        use_best_model=True,
        verbose=False,
        grow_policy='Depthwise',
        random_seed=random_state
    )
    model = cb.CatBoostClassifier(
        **params,
    )

    eval_set = cb.Pool(data=X_val, label=y_val)
    model.fit(X_train, y_train, eval_set=eval_set, plot=False)
    train_score = accuracy_score(model.predict(X_train), y_train)

    val_score = accuracy_score(model.predict(X_val), y_val)
    test_score = accuracy_score(model.predict(X_test), y_test)

    return train_score, val_score, test_score
