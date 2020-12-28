from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from utils.preprocessing import data_load
from utils.fea_eng import lgbm_preprocessing, xgb_preprocessing
from utils.fea_eng import cat_preprocessing
import numpy as np


def lgbm_cv(
    num_leaves: int,
    max_depth: int,
    min_child_samples: int,
    subsample: float,
    colsample_bytree: float,
    max_bin: float,
    reg_alpha: float,
    reg_lambda: float,
) -> float:
    train, test, submission = data_load("../../data/")
    train_le, test_le, label = lgbm_preprocessing(train, test)

    X_train, X_test, y_train, y_test = train_test_split(
        train_le, label, test_size=0.34, random_state=91
    )

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
        random_state=91,
    )

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_micro")
    return np.mean(scores)


def xgb_cv(
    learning_rate: float,
    n_estimators: int,
    max_depth: int,
    subsample: float,
    gamma: float,
) -> float:
    train, test, submission = data_load("../../data/")
    train_ohe, test_ohe, label = xgb_preprocessing(train, test)

    X_train, X_test, y_train, y_test = train_test_split(
        train_ohe, label, test_size=0.34, random_state=91
    )
    model = XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=int(round(n_estimators)),
        max_depth=int(round(max_depth)),
        subsample=subsample,
        gamma=gamma,
        random_state=91,
    )

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_micro")
    return np.mean(scores)


def cat_cv(
    iterations: float,
    depth: float,
    learning_rate: float,
    random_strength: float,
    bagging_temperature: float,
    border_count: float,
    l2_leaf_reg: float,
    scale_pos_weight: float,
) -> float:
    train, test, submission = data_load("../../data/")
    train_cat, test_cat, label = cat_preprocessing(train, test)

    X_train, X_test, y_train, y_test = train_test_split(
        train_cat, label, test_size=0.34, random_state=91
    )

    model = CatBoostClassifier(
        iterations=int(round(iterations)),
        depth=int(round(depth)),
        learning_rate=max(min(learning_rate, 1), 0),
        random_strength=max(min(random_strength, 1), 0),
        bagging_temperature=max(min(bagging_temperature, 1), 0),
        border_count=max(min(border_count, 1), 0),
        l2_leaf_reg=max(min(l2_leaf_reg, 1), 0),
        scale_pos_weight=max(min(scale_pos_weight, 1), 0),
    )
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_micro")
    return np.mean(scores)
