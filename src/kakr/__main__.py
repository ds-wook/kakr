from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import numpy as np
import argparse
from optim.bayesian_train import xgb_cv
from optim.bayesian_optim import xgb_parameter
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
from model.kfold_model import kfold_model
from model.kfold_model import stratified_kfold_model

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

    # lgbm 분류기
    lgb_clf = LGBMClassifier(objective='binary', verbose=400, random_state=91)

    xgb_param_bounds = {
        'learning_rate': (0.001, 0.1),
        'n_estimators': (100, 1000),
        'max_depth': (3, 8),
        'subsample': (0.4, 1.0),
        'gamma': (0, 3)}
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
    lgb_preds = stratified_kfold_model(lgb_clf, 5, train_ohe, target, test_ohe)
    xgb_preds = kfold_model(xgb_clf, 5, train_ohe, target, test_ohe)

    y_preds = 0.6 * lgb_preds + 0.4 * xgb_preds
    submission['prediction'] = y_preds

    for ix, row in submission.iterrows():
        if row['prediction'] > 0.5:
            submission.loc[ix, 'prediction'] = 1
        else:
            submission.loc[ix, 'prediction'] = 0
    submission = submission.astype({'prediction': np.int64})
    submission.to_csv(args.submit + args.file, index=False)
