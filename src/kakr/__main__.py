from lightgbm import LGBMClassifier
import numpy as np
import argparse
from utils.fea_eng import data_load
from utils.fea_eng import target_astype
from utils.fea_eng import drop_target
from utils.fea_eng import ordinal_encoder
from model.kfold_model import stratified_kfold_model
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    parse = argparse.ArgumentParser('Baseline Modeling')
    parse.add_argument('--path', type=str,
                       help='Input data load path',
                       default='../../data')

    parse.add_argument('--submit', type=str,
                       help='save the submit csv file',
                       default='../../res/')

    parse.add_argument('--file', type=str,
                       help='naming file name',
                       default='submission.csv')

    args = parse.parse_args()

    train, test, submission = data_load(args.path)
    target = target_astype(train)
    train, test = drop_target(train, test)
    train_le, test_le = ordinal_encoder(train, target, test)

    # lgbm 분류기
    lgb_clf = LGBMClassifier(objective='binary', verbose=400, random_state=91)

    lgb_preds1 = stratified_kfold_model(lgb_clf, 5, train_le, target, test_le)
    lgb_preds2 = stratified_kfold_model(lgb_clf, 5, train_le, target, test_le)

    y_preds = 0.6 * lgb_preds1 + 0.4 * lgb_preds2

    submission['prediction'] = y_preds

    for ix, row in submission.iterrows():
        if row['prediction'] > 0.5:
            submission.loc[ix, 'prediction'] = 1
        else:
            submission.loc[ix, 'prediction'] = 0
    submission = submission.astype({'prediction': np.int64})
    submission.to_csv(args.submit + args.file, index=False)
