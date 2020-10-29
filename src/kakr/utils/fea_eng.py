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


def education_map(
        train: pd.DataFrame,
        test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grouped = train.groupby(['education'])['income'].agg(['mean', 'count'])
    grouped = grouped.sort_values('mean').reset_index()
    edu_col = grouped['education'].values.tolist()
    lev_col = [f'level_{i}' for i in range(10)]
    lev_col += ['level_1', 'level_2', 'level_3', 'level_3',
                'level_6', 'level_9']
    lev_col = sorted(lev_col)
    education_map = {edu: lev for edu, lev in zip(edu_col, lev_col)}
    train['education'] = train['education'].map(education_map)
    test['education'] = test['education'].map(education_map)

    return train, test
