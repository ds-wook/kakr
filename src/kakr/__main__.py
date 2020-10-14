from lightgbm import LGBMClassifier
from category_encoders.ordinal import OrdinalEncoder
import numpy as np
import argparse
from utils.fea_eng import data_load


if __name__ == "__main__":
    parse = argparse.ArgumentParser('Baseline Modeling')
    parse.add_argument('--path', type=str,
                       help='Input data load path',
                       default='../../data')
    args = parse.parse_args()

    train, test, submission = data_load(args.path)
    target = train['income'] != '<=50K'
    train = train.drop('income', axis=1)
    le_encoder = OrdinalEncoder(list(train.columns))
    train_le = le_encoder.fit_transform(train, target)
    test_le = le_encoder.transform(test)

    lgb_model = LGBMClassifier(
                n_jobs=-1,
                n_estimators=1000,
                learning_rate=0.02,
                num_leaves=32,
                subsample=0.8,
                max_depth=12,
                silent=-1,
                verbose=-1)

    lgb_model.fit(train_le, target)
    lgb_pred = lgb_model.predict(test_le).astype(np.int64)
    submission['prediction'] = lgb_pred
    submission.to_csv('../../res/baseline_lgb_model.csv', index=False)
