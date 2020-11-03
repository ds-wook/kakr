from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from typing import Any


def kfold_model(
        model: Any,
        n_fold: int,
        train: pd.DataFrame,
        target: pd.Series,
        test: pd.DataFrame) -> np.ndarray:
    folds = KFold(n_splits=n_fold)
    splits = folds.split(train, target)
    y_preds = np.zeros(test.shape[0])

    for fold_n, (train_index, valid_index) in enumerate(splits):
        print(f'{model.__class__.__name__} Learning Start!')
        print(f'Fold: {fold_n + 1}')
        X_train, X_valid = train.iloc[train_index], train.iloc[valid_index]
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]

        evals = [(X_train, y_train), (X_valid, y_valid)]

        model.fit(X_train, y_train,
                  eval_set=evals, verbose=True)

        y_preds += model.predict(test).astype(np.int64) / n_fold

        del X_train, X_valid, y_train, y_valid

    return y_preds


def stratified_kfold_model(
        model: Any,
        n_fold: int,
        train: pd.DataFrame,
        target: pd.Series,
        test: pd.DataFrame) -> np.ndarray:
    folds = StratifiedKFold(n_splits=n_fold)
    splits = folds.split(train, target)
    y_preds = np.zeros(test.shape[0])

    for fold_n, (train_index, valid_index) in enumerate(splits):
        print(f'{model.__class__.__name__} Learning Start!')
        print(f'Fold: {fold_n + 1}')
        X_train, X_valid = train.iloc[train_index], train.iloc[valid_index]
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]

        evals = [(X_train, y_train), (X_valid, y_valid)]

        model.fit(X_train, y_train, eval_set=evals, verbose=True)

        y_preds += model.predict(test).astype(np.int64) / n_fold

        del X_train, X_valid, y_train, y_valid

    return y_preds
