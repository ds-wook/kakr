from bayes_opt import BayesianOptimization
from optim.bayesian_optim import lgbm_cv
from optim.bayesian_optim import xgb_cv
# 입력값의 탐색 대상 구간
lgb_param_bounds = {
        'learning_rate': (0.0001, 0.05),
        'num_leaves': (300, 600),
        'max_depth': (2, 25),
        'min_child_weight': (30, 100),
        'colsample_bytree': (0, 0.99),
        'feature_fraction': (0.0001, 0.99),
        'bagging_fraction': (0.0001, 0.99),
        'reg_alpha': (0, 1),
        'reg_lambda': (0, 1)}

lgbm_bo = BayesianOptimization(f=lgbm_cv, pbounds=lgb_param_bounds,
                               verbose=2, random_state=91)
lgbm_bo.maximize(init_points=5, n_iter=50)

print(f'LGBM Bayesian \n {lgbm_bo.max}')

xgb_param_bounds = {
        'learning_rate': (0.001, 0.1),
        'n_estimators': (100, 1000),
        'max_depth': (3, 8),
        'subsample': (0.4, 1.0),
        'gamma': (0, 3)}

xgb_bo = BayesianOptimization(f=xgb_cv, pbounds=xgb_param_bounds,
                              verbose=2, random_state=91)
xgb_bo.maximize(init_points=5, n_iter=50)

print(f'XGB Bayesian \n {xgb_bo.max}')
