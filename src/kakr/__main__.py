from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import argparse
from utils.fea_eng import data_load
from utils.fea_eng import target_astype
from utils.fea_eng import drop_target
from utils.fea_eng import ordinal_encoder
# from utils.evaluation import get_clf_eval
from model.stacking import get_stacking_base_datasets
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    parse = argparse.ArgumentParser('Baseline Modeling')
    parse.add_argument('--path', type=str,
                       help='Input data load path',
                       default='../../data')

    parse.add_argument('--submit', type=str,
                       help='save the submit csv file',
                       default='../../res')

    args = parse.parse_args()

    train, test, submission = data_load(args.path)
    target = target_astype(train)
    train = drop_target(train)
    train_le, test_le = ordinal_encoder(train, target, test)
    X_train, X_test, y_train, y_test =\
        train_test_split(train_le, target, test_size=0.2, random_state=2020)

    # lgbm 분류기
    lgb_clf = LGBMClassifier(
                n_jobs=-1,
                n_estimators=1000,
                random_state=2020,
                learning_rate=0.02,
                num_leaves=32,
                subsample=0.8,
                max_depth=12,
                silent=-1,
                verbose=-1)

    # xgb 분류기
    xgb_clf = XGBClassifier(
                n_jobs=-1,
                n_estimators=1000,
                learning_rate=0.02,
                random_state=2020)

    # cat 분류기
    cat_clf = CatBoostClassifier()

    # random forest 분류기
    rf_clf = RandomForestClassifier(
                n_jobs=-1,
                n_estimators=1000,
                max_depth=12,
                random_state=2020,
                verbose=-1)

    # logistic 분류기
    ada_clf = AdaBoostClassifier(n_estimators=1000)

    '''
    xgb_train, xgb_test =\
        get_stacking_base_datasets(xgb_clf, X_train, y_train, X_test, 5)

    cat_train, cat_test =\
        get_stacking_base_datasets(cat_clf, X_train, y_train, X_test, 5)

    rf_train, rf_test =\
        get_stacking_base_datasets(rf_clf, X_train, y_train, X_test, 5)

    ada_train, ada_test =\
        get_stacking_base_datasets(ada_clf, X_train, y_train, X_test, 5)

    stacking_final_X_train =\
        np.concatenate([rf_train, xgb_train, cat_train, ada_train], axis=1)

    stacking_final_X_test =\
        np.concatenate([rf_test, xgb_test, cat_test, ada_test], axis=1)

    lgb_clf.fit(stacking_final_X_train, y_train,
                eval_set=[(stacking_final_X_train, y_train),
                          (stacking_final_X_test, y_test)],
                verbose=100,
                early_stopping_rounds=100)

    stacking_final = lgb_clf.predict(stacking_final_X_test)

    get_clf_eval(y_test, stacking_final)
    '''

    xgb_train, xgb_test =\
        get_stacking_base_datasets(xgb_clf, train_le, target, test_le, 5)

    cat_train, cat_test =\
        get_stacking_base_datasets(cat_clf, train_le, target, test_le, 5)

    rf_train, rf_test =\
        get_stacking_base_datasets(rf_clf, train_le, target, test_le, 5)

    ada_train, ada_test =\
        get_stacking_base_datasets(ada_clf, train_le, target, test_le, 5)

    stacking_final_X_train =\
        np.concatenate([rf_train, xgb_train, cat_train, ada_train], axis=1)

    stacking_final_X_test =\
        np.concatenate([rf_test, xgb_test, cat_test, ada_test], axis=1)

    lgb_clf.fit(stacking_final_X_train, target)

    stacking_final = lgb_clf.predict(stacking_final_X_test)

    submission['prediction'] = stacking_final
    submission.to_csv(args.submit + '/stacking_model02.csv', index=False)
