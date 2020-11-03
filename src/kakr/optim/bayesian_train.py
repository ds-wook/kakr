from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from utils.preprocessing import data_load
from utils.preprocessing import concat_data
from utils.preprocessing import other_workclass
from utils.preprocessing import fnlwgt_log
from utils.preprocessing import education_map
from utils.preprocessing import marital_status_data
from utils.preprocessing import occupation_data
from utils.preprocessing import capital_net_data
from utils.preprocessing import convert_country_data
from utils.preprocessing import delete_column
from utils.preprocessing import ohe_data
from utils.preprocessing import split_data
from sklearn.model_selection import train_test_split

train, test, submission = data_load('../../data/')
all_data = concat_data(train, test)
all_data = other_workclass(all_data)
all_data = fnlwgt_log(all_data)
all_data = education_map(all_data)
all_data = marital_status_data(all_data)
all_data = occupation_data(all_data)
all_data = capital_net_data(all_data)
all_data = convert_country_data(all_data)
all_data = delete_column(all_data)
all_data_ohe = ohe_data(all_data)
train_ohe, test_ohe, target = split_data(all_data_ohe, train)

X_train, X_test, y_train, y_test =\
    train_test_split(train_ohe, target, test_size=0.2, random_state=91)


def lgbm_cv(
        num_leaves: int,
        max_depth: int,
        min_child_samples: int,
        subsample: float,
        colsample_bytree: float,
        max_bin: float,
        reg_alpha: float,
        reg_lambda: float) -> float:

    model = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.02,
                num_leaves=int(round(num_leaves)),
                max_depth=int(round(max_depth)),
                min_child_samples=int(round(min_child_samples)),
                subsample=max(min(subsample, 1), 0),
                colsample_bytree=max(min(colsample_bytree, 1), 0),
                max_bin=max(int(round(max_bin)), 10),
                reg_alpha=max(reg_alpha, 0),
                reg_lambda=max(reg_lambda, 0),
                random_state=91
            )

    scoring = {'f1_score': make_scorer(f1_score)}
    result = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
    f1 = result['test_f1_score'].mean()
    return f1


def xgb_cv(
        learning_rate: float,
        n_estimators: int,
        max_depth: int,
        subsample: float,
        gamma: float) -> float:

    model = XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=int(round(n_estimators)),
            max_depth=int(round(max_depth)),
            subsample=subsample,
            gamma=gamma,
            random_state=91)

    scoring = {'f1_score': make_scorer(f1_score)}
    result = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
    f1 = result['test_f1_score'].mean()
    return f1
