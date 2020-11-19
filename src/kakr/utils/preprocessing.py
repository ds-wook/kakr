import numpy as np
import pandas as pd
from typing import Tuple


def data_load(
        path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(path + 'train.csv')
    test = pd.read_csv(path + 'test.csv')
    submission = pd.read_csv(path + 'sample_submission.csv')
    return train, test, submission


def education_map(
        all_data: pd.DataFrame) -> pd.DataFrame:
    grouped = all_data.groupby('education')['income'].agg(['mean', 'count'])
    grouped = grouped.sort_values('mean').reset_index()
    edu_col = grouped['education'].values.tolist()
    lev_col = [f'level_{i}' for i in range(10)]
    lev_col += ['level_1', 'level_2', 'level_3',
                'level_3', 'level_6', 'level_9']
    lev_col.sort()
    education_map = {edu: lev for edu, lev in zip(edu_col, lev_col)}
    all_data['education'] = all_data['education'].map(education_map)
    all_data = all_data.drop('education_num', axis=1)
    return all_data


def capital_net_data(
        all_data: pd.DataFrame) -> pd.DataFrame:
    all_data['capital_net'] =\
        all_data['capital_gain'] - all_data['capital_loss']
    pos_key = all_data.loc[(all_data['income'] == 1)
                           & (all_data['capital_net'] > 0),
                           'capital_net'].value_counts().sort_index().keys()
    pos_key = pos_key.tolist()

    neg_key = all_data.loc[(all_data['income'] == 0)
                           & (all_data['capital_net'] > 0),
                           'capital_net'].value_counts().sort_index().keys()
    neg_key = neg_key.tolist()
    capital_net_pos_key = [key for key in pos_key if key not in neg_key]
    capital_net_neg_key = [key for key in neg_key if key not in pos_key]
    all_data['capital_net_pos_key'] =\
        all_data['capital_net'].apply(lambda x: x in capital_net_pos_key)
    all_data['capital_net_neg_key'] =\
        all_data['capital_net'].apply(lambda x: x in capital_net_neg_key)

    return all_data


def split_data(
        all_data_ohe: pd.DataFrame,
        train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train_features =\
        all_data_ohe.drop('income', axis=1).iloc[:train.shape[0]]
    test_features =\
        all_data_ohe.drop('income', axis=1).iloc[train.shape[0]:]

    target = all_data_ohe['income'].iloc[:train.shape[0]]
    target = target.astype(np.int64)
    return train_features, test_features, target
