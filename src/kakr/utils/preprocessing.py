import numpy as np
import pandas as pd
from typing import Tuple


def data_load(
        path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(path + 'train.csv')
    test = pd.read_csv(path + 'test.csv')
    submission = pd.read_csv(path + 'sample_submission.csv')
    return train, test, submission


def concat_data(
        train: pd.DataFrame,
        test: pd.DataFrame) -> pd.DataFrame:
    train['income'] = train['income'].apply(lambda x: 0 if x == '<=50K' else 1)
    all_data = pd.concat([train, test], sort=False)
    return all_data


def other_workclass(
        all_data: pd.DataFrame) -> pd.DataFrame:
    others = ['Without-pay', 'Never-worked']
    all_data['workclass'] = all_data['workclass'].apply(lambda x: 'Other'
                                                        if x in others else x)
    return all_data


def fnlwgt_log(
        all_data: pd.DataFrame) -> pd.DataFrame:
    all_data['fnlwgt_log'] = np.log(all_data['fnlwgt'])
    return all_data


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


def marital_status_data(
        all_data: pd.DataFrame) -> pd.DataFrame:
    all_data.loc[all_data['marital_status'] == 'Married-AF-spouse',
                 'marital_status'] = 'Married-civ-spouse'
    return all_data


def occupation_data(
        all_data: pd.DataFrame) -> pd.DataFrame:
    all_data.loc[all_data['occupation'].isin(['Armed-Forces', 'Priv-house-serv']),
                 'occupation'] = 'Priv-house-serv'
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


def convert_country(x: object) -> None:
    income_01 = ['Jamaica', 'Haiti', 'Puerto-Rico',
                 'Laos', 'Thailand', 'Ecuador']
    income_02 = ['Outlying-US(Guam-USVI-etc)', 'Honduras', 'Columbia',
                 'Dominican-Republic', 'Mexico', 'Guatemala', 'Portugal',
                 'Trinadad&Tobago', 'Nicaragua', 'Peru',
                 'Vietnam', 'El-Salvador']
    income_03 = ['Poland', 'Ireland', 'South', 'China']
    income_04 = ['United-States']
    income_05 = ['Greece', 'Scotland', 'Cuba', 'Hungary',
                 'Hong', 'Holand-Netherlands']
    income_06 = ['Philippines', 'Canada']
    income_07 = ['England', 'Germany']
    income_08 = ['Italy', 'India', 'Japan', 'France', 'Yugoslavia', 'Cambodia']
    income_09 = ['Taiwan', 'Iran']

    if x in income_01:
        return 'income_01'
    elif x in income_02:
        return 'income_02'
    elif x in income_03:
        return 'income_03'
    elif x in income_04:
        return 'income_04'
    elif x in income_05:
        return 'income_05'
    elif x in income_06:
        return 'income_06'
    elif x in income_07:
        return 'income_07'
    elif x in income_08:
        return 'income_08'
    elif x in income_09:
        return 'income_09'
    else:
        return 'income_other'


def convert_country_data(
        all_data: pd.DataFrame) -> pd.DataFrame:
    all_data['country_bin'] =\
        all_data['native_country'].apply(convert_country)
    return all_data


def delete_column(
        all_data: pd.DataFrame) -> pd.DataFrame:
    all_data =\
        all_data.drop(['id', 'fnlwgt', 'capital_net'], axis=1)
    return all_data


def ohe_data(
        all_data: pd.DataFrame) -> pd.DataFrame:
    all_data_ohe = pd.get_dummies(all_data)
    return all_data_ohe


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
