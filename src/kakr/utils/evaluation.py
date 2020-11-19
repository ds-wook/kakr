from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np


def get_clf_eval(
        y_true: np.ndarray,
        y_pred: np.ndarray) -> None:
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='micro')
    print(f'Precision: {precision:.5f}', end=' ')
    print(f'Recall: {recall:.5f}, F1 score: {f1:.8f}')
