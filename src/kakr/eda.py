# %%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# %% [markdown]
'''
### train test load
'''

train = pd.read_csv('../../data/train.csv')
test = pd.read_csv('../../data/test.csv')


# %%


print(train.shape)
train.head()


# %%


print(test.shape)
test.head()


# %% [markdown]


'''
## 데이터 세부 설명
### train/test는 14개의 columns으로 구성되어 있고, train은 예측해야 하는 target 값 feature까지 1개가 추가로 있습니다. 각 데이터는 다음을 의미합니다.

+ id
+ age : 나이
+ workclass : 고용 형태
+ fnlwgt : 사람 대표성을 나타내는 가중치 (final weight의 약자)
+ education : 교육 수준
+ education_num : 교육 수준 수치
+ marital_status: 결혼 상태
+ occupation : 업종
+ relationship : 가족 관계
+ race : 인종
+ sex : 성별
+ capital_gain : 양도 소득
+ capital_loss : 양도 손실
+ hours_per_week : 주당 근무 시간
+ native_country : 국적
+ income : 수익 (예측해야 하는 값)
  >50K : 1
  <=50K : 0
'''
train.info()


# %%

train['income'] = train['income'].apply(lambda x: 0 if x == '<=50K' else 1)
train['income'].value_counts()


# %%

train['income'] = train['income']


# %%

all_data = pd.concat([train, test], sort=False)


# %% [markdown]
'''
### workclass: 직장
'''
# %%
all_data['workclass'].value_counts()
# %%
grouped = all_data.groupby('workclass')['income'].mean().sort_values()
grouped = grouped.reset_index()
sns.barplot(x='workclass', y='income', data=grouped)
plt.xticks(rotation=30)
plt.show()
# %%
all_data.groupby('workclass')['income'].mean().to_dict()
# %%
workclass_other = ['Without-pay', 'Never-worked']
all_data['workclass'] = all_data['workclass'].apply(lambda x: 'Other' if x in workclass_other else x)

# %%
all_data['workclass'].value_counts()
# %% [markdown]
'''
### age: 나이
'''
income0 = all_data.loc[all_data['income'] == 0, 'age']
income1 = all_data.loc[all_data['income'] == 1, 'age']

plt.figure(figsize=(10, 7))
sns.distplot(income0, kde=True, rug=True, hist=False, color='blue')
sns.distplot(income1, kde=True, rug=True, hist=False, color='red')
plt.show()
# %%
sns.distplot(np.log(all_data['fnlwgt']))
# %%
all_data['fnlwgt_log'] = np.log(all_data['fnlwgt'])
# %% [markdown]
'''
### education: 교육
'''
all_data['education'].value_counts()
# %%[markdown]
'''
#### 학교를 안나온 사람들에 대한 수입 통계
'''
all_data[all_data['education'] == 'Preschool']['income'].sum()

# %% [markdown]
'''
#### 교육에 대한 수입 평균
'''

grouped = all_data.groupby('education')['income'].agg(['mean', 'count'])
grouped = grouped.sort_values('mean').reset_index()
grouped


# %%


edu_col = grouped['education'].values.tolist()
edu_col


# %%
lev_col = [f'level_{i}' for i in range(10)]
lev_col += ['level_1', 'level_2', 'level_3', 'level_3', 'level_6', 'level_9']
lev_col.sort()
lev_col
# %%
education_map = {edu: lev for edu, lev in zip(edu_col, lev_col)}
all_data['education'] = all_data['education'].map(education_map)
all_data['education'].value_counts()
# %%
all_data.drop('education_num', axis=1, inplace=True)
# %%
all_data.columns
# %% [markdown]


'''
#####marital_status: 결혼 상태
'''
all_data['marital_status'].value_counts()


# %%


all_data.groupby(['marital_status'])['income'].agg(['mean', 'count'])


# %%

# 배우자에 의한 결혼의 비율을 같은 feature로 묶어줌

all_data.loc[all_data['marital_status'] == 'Married-AF-spouse',
             'marital_status'] = 'Married-civ-spouse'
all_data['marital_status'].value_counts()


# %%[markdown]
'''
### occupation: 직업
'''

all_data['occupation'].value_counts()


# %%

all_data.loc[train['occupation'].isin(['Armed-Forces', 'Priv-house-serv']),
             'income'].value_counts()


# %% [markdown]
'''
### 거의 없는 직업군끼리는 서로 묶어준다.
'''

all_data.loc[all_data['occupation'].isin(['Armed-Forces', 'Priv-house-serv']),
             'occupation'] = 'Priv-house-serv'


# %%
train.dtypes[train.dtypes == 'object'].index
# %%
all_data.groupby('workclass')['income'].mean().to_dict()['income']
# %%
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from optim.bayesian_test import lgbm_cv, xgb_cv, cat_cv
from optim.bayesian_optim import lgbm_parameter, xgb_parameter, cat_parameter
from utils.preprocessing import data_load
from utils.submission import submit_file
from utils.fea_eng import lgbm_preprocessing, xgb_preprocessing
from model.kfold_model import stratified_kfold_model
from model.kfold_model import voting_kfold_model


train, test, submission = data_load('../../data/')
train_ohe, test_ohe, label = lgbm_preprocessing(train, test)
print(f'train shape: {train_ohe.shape}')
print(f'test shape: {test_ohe.shape}')

cat_param_bounds = {
    'iterations': (10, 1000),
    'depth': (1, 8),
    'learning_rate': (0.01, 1),
    'random_strength': (0.01, 10),
    'bagging_temperature': (0.0, 1.0),
    'border_count': (1, 255),
    'l2_leaf_reg': (2, 30),
    'scale_pos_weight': (0.01, 1)
}

bo_cat = cat_parameter(cat_cv, cat_param_bounds)

print(bo_cat)
# %%
