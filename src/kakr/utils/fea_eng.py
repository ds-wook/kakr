import pandas as pd
from typing import Tuple


def data_load(
        path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(path + '/train.csv')
    test = pd.read_csv(path + '/test.csv')
    submission = pd.read_csv(path + '/sample_submission.csv')
    return train, test, submission
