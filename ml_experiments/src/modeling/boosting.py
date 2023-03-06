from typing import Tuple

import catboost as cb
from sklearn.metrics import accuracy_score
import optuna
import scipy
import numpy as np


def catboosting(X_train, y_train, X_val, y_val, X_test, y_test, random_state: int = 12) -> Tuple[float, float, float]:
    params = {
        'max_depth': optuna.distributions.IntDistribution(1, 12),
        'learning_rate': optuna.distributions.FloatDistribution(0.01, 1.0, log=True),
        'n_estimators': optuna.distributions.IntDistribution(50, 1000),
        'subsample': optuna.distributions.FloatDistribution(0.02, 1.0, log=True),
        'colsample_bylevel': optuna.distributions.FloatDistribution(0.01, 1.0, log=True),
        'bagging_temperature': optuna.distributions.FloatDistribution(1e-8, 1.0, log=True),
        'l2_leaf_reg': optuna.distributions.FloatDistribution(1e-8, 100, log=True),
        'min_child_samples': optuna.distributions.IntDistribution(1, 10)
    }

    model = cb.CatBoostClassifier(verbose=False)

    if scipy.sparse.issparse(X_train):
        X_train = scipy.sparse.vstack((X_train, X_val))
        y_train = scipy.sparse.hstack((y_train, y_val))
    else:
        X_train = np.vstack((X_train, X_val))
        y_train = np.hstack((y_train, y_val))

    optuna_search = optuna.integration.OptunaSearchCV(model, params, n_trials=100, random_state=12)
    optuna_search.fit(X_train, y_train)

    model = cb.CatBoostClassifier(**optuna_search.best_params_)
    model.fit(X_train, y_train)

    train_score = accuracy_score(model.predict(X_train), y_train)
    val_score = accuracy_score(model.predict(X_val), y_val)
    test_score = accuracy_score(model.predict(X_test), y_test)

    return train_score, val_score, test_score
