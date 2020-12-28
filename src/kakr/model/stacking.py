import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from typing import Any, Tuple


def get_stacking_base_datasets(
    model: Any,
    X_train_n: pd.DataFrame,
    y_train_n: pd.Series,
    X_test_n: pd.DataFrame,
    n_folds: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # 지정된 n_folds값으로 kfold 생성
    kf = KFold(n_splits=n_folds, shuffle=False, random_state=2020)
    # 추후에 메타 모델이 사용할 학습 데이터 반환을 위한 넘파이 배열 초기화
    train_fold_pred = np.zeros((X_train_n.shape[0], 1))
    test_pred = np.zeros((X_test_n.shape[0], n_folds))
    print(f"{model.__class__.__name__} model training!")

    for fold_cnt, (train_idx, valid_idx) in enumerate(kf.split(X_train_n)):
        # 입력된 핛브 데이터에서 기반 모델이 학습/예측할 폴드 데이터 세트 추출
        print(f"\t fold set: {fold_cnt + 1} Start!")
        X_tr = X_train_n.iloc[train_idx]
        y_tr = y_train_n.iloc[train_idx]
        X_te = X_train_n.iloc[valid_idx]
        # 폴드 세트 내부에서 다시 만들어진 학습데이터로 기반 모델의 학습 수행
        model.fit(X_tr, y_tr)
        # 폴드 세트 내부에서 다시 만들어진 검증 데이터로 기반 모델 예측 후 데이터 저장
        train_fold_pred[valid_idx, :] = model.predict(X_te).reshape(-1, 1)
        # 입력된 원본 테스트 데이터를 폴드 세트내 학습된 기반 모델에서 예측 후 데이터 저장
        test_pred[:, fold_cnt] = model.predict(X_test_n)

    # 폴드 세트 내에서 원본 테스트 데이터를 예측한 데이터를 평균하여 테스트 데이터로 생성
    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1, 1)

    # train_fold_pred는 최종 메타 모델이 사용하는 학습 데이터, test_pred_mean은 테스트 데이터
    return train_fold_pred, test_pred_mean
