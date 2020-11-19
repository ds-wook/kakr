import numpy as np
import pandas as pd
from typing import Tuple


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


def xgb_preprocessing(
        train: pd.DataFrame,
        test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train['income'] = train['income'].map(lambda x: 1 if x == '>50K' else 0)
    label = train['income']
    del train['id']
    del test['id']
    # 결측치 제거
    has_na_columns = ['workclass', 'occupation', 'native_country']

    for col in has_na_columns:
        train.loc[train[col] == '?', col] = train[col].mode()[0]
        test.loc[test[col] == '?', col] = test[col].mode()[0]

    # log
    train['log_capital_gain'] =\
        train['capital_gain'].map(lambda x: np.log(x) if x != 0 else 0)
    test['log_capital_gain'] =\
        test['capital_gain'].map(lambda x: np.log(x) if x != 0 else 0)

    train['log_capital_loss'] =\
        train['capital_loss'].map(lambda x: np.log(x) if x != 0 else 0)
    test['log_capital_loss'] =\
        test['capital_loss'].map(lambda x: np.log(x) if x != 0 else 0)

    train.drop(['capital_gain', 'capital_loss'], axis=1, inplace=True)
    test.drop(['capital_gain', 'capital_loss'], axis=1, inplace=True)

    all_data = pd.concat([train, test])

    others = ['Without-pay', 'Never-worked']
    all_data['workclass'] = all_data['workclass'].apply(lambda x: 'Other'
                                                        if x in others else x)
    all_data['country_bin'] =\
        all_data['native_country'].apply(convert_country)

    all_data['fnlwgt_log'] = np.log(all_data['fnlwgt'])

    all_data.loc[all_data['marital_status'] == 'Married-AF-spouse',
                 'marital_status'] = 'Married-civ-spouse'
    all_data.loc[
        all_data['occupation'].isin(['Armed-Forces', 'Priv-house-serv']),
        'occupation'] = 'Priv-house-serv'

    cat_col = ['workclass', 'marital_status', 'occupation',
               'relationship', 'race', 'sex', 'native_country']

    for col in cat_col:
        map_dic = train.groupby(col)['income'].mean()
        map_dic = map_dic.to_dict()
        all_data[col + '_mean_enc'] =\
            all_data[col].map(lambda x: map_dic.get(x))
    all_data.drop(['income', 'fnlwgt', 'education_num'], axis=1, inplace=True)
    all_data_ohe = pd.get_dummies(all_data)
    train = all_data_ohe.iloc[:len(train)]
    test = all_data_ohe.iloc[len(train):]

    return train, test, label


def lgbm_preprocessing(
        train: pd.DataFrame,
        test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train['income'] = train['income'].map(lambda x: 1 if x == '>50K' else 0)
    label = train['income']
    del train['id']
    del test['id']
    # 결측치 제거
    has_na_columns = ['workclass', 'occupation', 'native_country']

    for col in has_na_columns:
        train.loc[train[col] == '?', col] = train[col].mode()[0]
        test.loc[test[col] == '?', col] = test[col].mode()[0]

    # log
    train['log_capital_gain'] =\
        train['capital_gain'].map(lambda x: np.log(x) if x != 0 else 0)
    test['log_capital_gain'] =\
        test['capital_gain'].map(lambda x: np.log(x) if x != 0 else 0)

    train['log_capital_loss'] =\
        train['capital_loss'].map(lambda x: np.log(x) if x != 0 else 0)
    test['log_capital_loss'] =\
        test['capital_loss'].map(lambda x: np.log(x) if x != 0 else 0)

    all_data = pd.concat([train, test])

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

    others = ['Without-pay', 'Never-worked']

    all_data['workclass'] = all_data['workclass'].apply(lambda x: 'Other'
                                                        if x in others else x)
    all_data['country_bin'] =\
        all_data['native_country'].apply(convert_country)

    all_data['fnlwgt_log'] = np.log(all_data['fnlwgt'])

    all_data.loc[all_data['marital_status'] == 'Married-AF-spouse',
                 'marital_status'] = 'Married-civ-spouse'
    all_data.loc[
        all_data['occupation'].isin(['Armed-Forces', 'Priv-house-serv']),
        'occupation'] = 'Priv-house-serv'

    cat_col = ['workclass', 'marital_status', 'occupation',
               'relationship', 'race', 'sex', 'native_country']

    for col in cat_col:
        map_dic = train.groupby(col)['income'].mean()
        map_dic = map_dic.to_dict()
        all_data[col + '_mean_enc'] =\
            all_data[col].map(lambda x: map_dic.get(x))
    all_data.drop(['income', 'fnlwgt', 'education_num'], axis=1, inplace=True)
    all_data_ohe = pd.get_dummies(all_data)
    train = all_data_ohe.iloc[:len(train)]
    test = all_data_ohe.iloc[len(train):]

    return train, test, label
