# %%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualize import show_count_by_target
from utils.visualize import show_hist_by_target
import warnings
warnings.filterwarnings('ignore')


# %%


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


# %%


train.info()


# %%


train['income'].value_counts()


# %%

object_columns = train.dtypes[train.dtypes == 'object'].index.tolist()
object_columns = [col for col in object_columns if col not in ['income']]
print(object_columns)


# %%


show_count_by_target(train, object_columns)


# %%


num_columns = train.dtypes[train.dtypes != 'object'].index.tolist()
num_columns = [col for col in num_columns if col not in ['id']]
print(num_columns)


# %%


show_hist_by_target(train, num_columns)


# %%


sns.boxplot(x='education_num', data=train)
plt.show()
# %%


from category_encoders.ordinal import OrdinalEncoder
target = train['income'] != '<=50K'
train.drop(['income'], axis=1, inplace=True)
le_encoder = OrdinalEncoder(list(train.columns))
train_le = le_encoder.fit_transform(train, target)
test_le = le_encoder.transform(test)

train_le.head()
# %%
train.head()
# %%
for col in object_columns:
    train[col] = pd.factorize(train[col])[0]
    test[col] = pd.factorize(test[col])[0]
train.head()
# %%
train_le['workclass'].value_counts()
# %%
train['workclass'].value_counts()
# %%
test_le.head()
# %%
sns.displot(target)
plt.show()
# %%
test.head()
# %%
test_le['workclass'].value_counts()
# %%
test['workclass'].value_counts()
# %%
from lightgbm import LGBMClassifier

lgb_model = LGBMClassifier(
                n_jobs=-1,
                n_estimators=1000,
                learning_rate=0.02,
                num_leaves=32,
                subsample=0.8,
                max_depth=12,
                silent=-1,
                verbose=-1)
lgb_model.fit(train_le, target)
lgb_pred = lgb_model.predict(test_le).astype(np.int64)
submission_test = pd.read_csv('../../data/sample_submission.csv')
submission_test['predict'] = lgb_pred
submission_test['predict'].value_counts()
# %%
lgb_model.fit(train, target)
lgb_pred = lgb_model.predict(test).astype(np.int64)
submission_test['predict'] = lgb_pred
submission_test['predict'].value_counts()
# %%
from catboost import CatBoostClassifier
cat_clf = CatBoostClassifier()
cat_clf.fit(train_le, target)
cat_pred = cat_clf.predict(test_le)
submission_test['predict'] = cat_pred

submission_test['predict'].value_counts()
# %%
submission_test['predict'] = submission_test['predict'].astype(np.bool)
# %%
submission_test['predict'] = submission_test['predict'].astype(np.int64)
# %%
