import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import gc
from typing import Any


def ensemble_kfold(
    model: Any,
    X_train_n: pd.DataFrame,
    y_train_n: pd.Series,
    X_test_n: pd.DataFrame,
    n_folds: int,
) -> np.ndarray:
    kf = KFold(n_splits=n_folds)
    split = kf.split(X_train_n, y_train_n)
    y_preds = np.zeros(X_test_n.shape[0])
    print(f"{model.__class__.__name__} model training!")

    for fold_n, (train_idx, test_idx) in enumerate(split):
        X_train, X_valid = X_train_n.iloc[train_idx], X_train_n.iloc[test_idx]
        y_train, y_valid = y_train_n.iloc[train_idx], y_train_n.iloc[test_idx]

        evals = [(X_train, y_train), (X_valid, y_valid)]
        model.fit(X_train, y_train, eval_set=evals, verbose=True)
        y_preds += model.predict_proba(X_test_n)[:, 1] / n_folds
        y_preds.reshape(-1, 1)
        del X_train, X_valid, y_train, y_valid
        gc.collect()
    return y_preds
