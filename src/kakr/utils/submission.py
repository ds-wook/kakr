import numpy as np
import pandas as pd


def submit_file(
        submission: pd.DataFrame,
        y_preds: np.ndarray,
        submit: str,
        files: str) -> pd.DataFrame:
    submission['prediction'] = y_preds
    for ix, row in submission.iterrows():
        if row['prediction'] > 0.5:
            submission.loc[ix, 'prediction'] = 1
        else:
            submission.loc[ix, 'prediction'] = 0
    submission = submission.astype({'prediction': np.int64})
    return submission
