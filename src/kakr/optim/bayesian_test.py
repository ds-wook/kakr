from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
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

    scores = cross_val_score(model, train_le, label, scoring="f1_micro", cv=5)
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

    model = XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=int(round(n_estimators)),
        max_depth=int(round(max_depth)),
        subsample=max(min(subsample, 1), 0),
        gamma=gamma,
        random_state=91,
    )

    scores = cross_val_score(model, train_ohe, label, scoring="f1_micro", cv=5)
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

    scores = cross_val_score(model, train_cat, label, scoring="f1_micro", cv=5)
    return np.mean(scores)
