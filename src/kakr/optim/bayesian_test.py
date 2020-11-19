from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from utils.preprocessing import data_load
from utils.fea_eng import lgbm_preprocessing, xgb_preprocessing


def lgbm_cv(
        num_leaves: int,
        max_depth: int,
        min_child_samples: int,
        subsample: float,
        colsample_bytree: float,
        max_bin: float,
        reg_alpha: float,
        reg_lambda: float) -> float:
    train, test, submission = data_load('../../data/')
    train_ohe, test_ohe, label = lgbm_preprocessing(train, test)

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
    result = cross_validate(model, train_ohe, label, cv=5, scoring=scoring)
    f1 = result['test_f1_score'].mean()
    return f1


def xgb_cv(
        learning_rate: float,
        n_estimators: int,
        max_depth: int,
        subsample: float,
        gamma: float) -> float:
    train, test, submission = data_load('../../data/')
    train_ohe, test_ohe, label = xgb_preprocessing(train, test)

    model = XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=int(round(n_estimators)),
            max_depth=int(round(max_depth)),
            subsample=max(min(subsample, 1), 0),
            gamma=gamma,
            random_state=91
        )

    scoring = {'f1_score': make_scorer(f1_score)}
    result = cross_validate(model, train_ohe, label, cv=5, scoring=scoring)
    f1 = result['test_f1_score'].mean()
    return f1
