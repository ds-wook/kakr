from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import argparse
from optim.bayesian_test import lgbm_cv, xgb_cv
from optim.bayesian_optim import lgbm_parameter, xgb_parameter
from utils.preprocessing import data_load
from utils.submission import submit_file
from utils.fea_eng import lgbm_preprocessing, xgb_preprocessing
from model.kfold_model import stratified_kfold_model, kfold_model


if __name__ == "__main__":
    parse = argparse.ArgumentParser('Baseline Modeling')
    parse.add_argument('--path', type=str,
                       help='Input data load path',
                       default='../../data/')

    parse.add_argument('--submit', type=str,
                       help='save the submit csv file',
                       default='../../res/')

    parse.add_argument('--file', type=str,
                       help='naming file name',
                       default='submission.csv')

    args = parse.parse_args()

    train, test, submission = data_load(args.path)
    train_ohe, test_ohe, label = lgbm_preprocessing(train, test)
    print(f'train shape: {train_ohe.shape}')
    print(f'test shape: {test_ohe.shape}')

    lgb_param_bounds = {
        'max_depth': (4, 10),
        'num_leaves': (24, 1024),
        'min_child_samples': (10, 200),
        'subsample': (0.5, 1),
        'colsample_bytree': (0.5, 1),
        'max_bin': (10, 500),
        'reg_lambda': (0.001, 10),
        'reg_alpha': (0.01, 50)
    }

    bo_lgb = lgbm_parameter(lgbm_cv, lgb_param_bounds)

    # lgbm 분류기
    lgb_clf = LGBMClassifier(
                objective='binary',
                verbose=400,
                random_state=91,
                n_estimators=500,
                learning_rate=0.02,
                max_depth=int(round(bo_lgb['max_depth'])),
                num_leaves=int(round(bo_lgb['num_leaves'])),
                min_child_samples=int(round(bo_lgb['min_child_samples'])),
                subsample=max(min(bo_lgb['subsample'], 1), 0),
                colsample_bytree=max(min(bo_lgb['colsample_bytree'], 1), 0),
                max_bin=max(int(round(bo_lgb['max_bin'])), 10),
                reg_lambda=max(bo_lgb['reg_lambda'], 0),
                reg_alpha=max(bo_lgb['reg_alpha'], 0)
                )
    lgb_preds = stratified_kfold_model(lgb_clf, 5, train_ohe, label, test_ohe)

    train, test, submission = data_load('../../data/')
    train_ohe, test_ohe, label = xgb_preprocessing(train, test)
    print(f'train shape: {train_ohe.shape}')
    print(f'test shape: {test_ohe.shape}')

    xgb_param_bounds = {
        'learning_rate': (0.001, 0.1),
        'n_estimators': (100, 1000),
        'max_depth': (3, 8),
        'subsample': (0.4, 1.0),
        'gamma': (0, 3)
    }

    bo_xgb = xgb_parameter(xgb_cv, xgb_param_bounds)
    # xgb 분류기
    xgb_clf = XGBClassifier(
                objective='binary:logistic',
                random_state=91,
                learning_rate=bo_xgb['learning_rate'],
                n_estimators=int(round(bo_xgb['n_estimators'])),
                max_depth=int(round(bo_xgb['max_depth'])),
                subsample=bo_xgb['subsample'],
                gamma=bo_xgb['gamma'])
    xgb_preds = kfold_model(xgb_clf, 5, train_ohe, label, test_ohe)
    y_preds = 0.5 * lgb_preds + 0.5 * xgb_preds
    submission = submit_file(submission, y_preds, args.submit, args.file)
