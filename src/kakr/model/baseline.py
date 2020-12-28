from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd


def lgb_model(
    train_le: pd.DataFrame, target: pd.Series, test_le: pd.DataFrame
) -> np.ndarray:
    lgb_model = LGBMClassifier(
        n_jobs=-1,
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=32,
        subsample=0.8,
        max_depth=12,
        silent=-1,
        verbose=-1,
    )

    lgb_model.fit(train_le, target)
    lgb_pred = lgb_model.predict(test_le)
    return lgb_pred


def xgb_model(
    train_le: pd.DataFrame, target: pd.Series, test_le: pd.DataFrame
) -> np.ndarray:
    xgb_model = XGBClassifier(
        n_estimators=1000, learning_rate=0.02, max_depth=12, n_jobs=-1
    )
    xgb_model.fit(train_le, target)
    xgb_pred = xgb_model.predict(test_le)
    return xgb_pred


def cat_mode(
    train_le: pd.DataFrame, target: pd.Series, test_le: pd.DataFrame
) -> np.ndarray:
    cat_model = CatBoostClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=12,
        num_leaves=32,
        subsample=12,
        silent=-1,
        verbose=-1,
    )
    cat_model.fit(train_le, target)
    cat_pred = cat_model.predict(test_le)
    return cat_pred
