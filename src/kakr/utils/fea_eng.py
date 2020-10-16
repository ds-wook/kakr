import numpy as np
import pandas as pd
from typing import Tuple
from category_encoders.ordinal import OrdinalEncoder


def data_load(
        path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(path + '/train.csv')
    test = pd.read_csv(path + '/test.csv')
    submission = pd.read_csv(path + '/sample_submission.csv')
    return train, test, submission


def target_astype(
        train: pd.DataFrame) -> pd.Series:
    target = train['income'] != '<=50K'
    target = target.astype(np.int64)
    return target


def drop_target(
        train: pd.DataFrame,
        test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train.drop(['id', 'income'], axis=1, inplace=True)
    test.drop(['id'], axis=1, inplace=True)
    return train, test


def ordinal_encoder(
        train: pd.DataFrame,
        target: pd.Series,
        test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    le_encoder = OrdinalEncoder(list(train.columns))
    train_le = le_encoder.fit_transform(train, target)
    test_le = le_encoder.transform(test)

    return train_le, test_le


def one_hot_encoder(
        train: pd.DataFrame,
        test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_oh = pd.get_dummies(train)
    test_oh = pd.get_dummies(test)
    return train_oh, test_oh
