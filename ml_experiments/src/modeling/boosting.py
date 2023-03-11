from typing import Tuple, Union, Any

import catboost as cb
from sklearn.metrics import accuracy_score


def catboosting(X_train, y_train, X_val, y_val, X_test, y_test, quality_metric: Union[Any, str] = 'default',
                random_state: int = 12) -> Tuple[float, float, float]:
    if quality_metric == 'default':
        quality_metric = accuracy_score

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
    train_score = quality_metric(model.predict(X_train), y_train)

    val_score = quality_metric(model.predict(X_val), y_val)
    test_score = quality_metric(model.predict(X_test), y_test)

    return train_score, val_score, test_score
