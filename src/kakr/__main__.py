from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from category_encoders import CatBoostEncoder
import argparse
from utils.fea_eng import data_load
from utils.fea_eng import target_astype
from utils.fea_eng import drop_target
from utils.fea_eng import ordinal_encoder


if __name__ == "__main__":
    parse = argparse.ArgumentParser('Baseline Modeling')
    parse.add_argument('--path', type=str,
                       help='Input data load path',
                       default='../../data')
    args = parse.parse_args()

    train, test, submission = data_load(args.path)
    target = target_astype(train)
    train = drop_target(train)
    # train_le, test_le = ordinal_encoder(train, target, test)
    cat_encoder = CatBoostEncoder(list(train.columns))
    train_cat = cat_encoder.fit_transform(train,  target)
    test_cat = cat_encoder.transform(test)
    X_train, X_test, y_train, y_test =\
        train_test_split(train_cat, target, test_size=0.2, random_state=2020)
    # lgb_model = LGBMClassifier(
    #                 n_jobs=-1,
    #                 n_estimators=1000,
    #                 learning_rate=0.02,
    #                 num_leaves=32,
    #                 subsample=0.8,
    #                 max_depth=12,
    #                 silent=-1,
    #                 verbose=-1)

    # lgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
    #               eval_metric='acc', verbose=100, early_stopping_rounds=100)
    # lgb_pred = lgb_model.predict(X_test)
    # print(f'LGBM ACC: {accuracy_score(y_test, lgb_pred)}')

    cat_model = CatBoostClassifier()
    cat_model.fit(X_train, y_train)
    cat_pred = cat_model.predict(X_test)
    print(f'Cat ACC: {accuracy_score(y_test, cat_pred)}')
