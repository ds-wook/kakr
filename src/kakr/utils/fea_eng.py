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
        train: pd.DataFrame) -> pd.DataFrame:
    train.drop('income', axis=1, inplace=True)
    return train


def ordinal_encoder(
        train: pd.DataFrame,
        target: pd.Series,
        test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    le_encoder = OrdinalEncoder(list(train.columns))
    train_le = le_encoder.fit_transform(train, target)
    test_le = le_encoder.transform(test)

    return train_le, test_le
