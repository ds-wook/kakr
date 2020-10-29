import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import seaborn as sns
from typing import List


def show_count_by_target(
        df: pd.DataFrame,
        columns: List[str]) -> None:
    '''Visualization count plot
    Args:
        df: train dataset
        columns: continue's columns
    '''
    for col in columns:
        plt.figure(figsize=(12, 4))
        chart = sns.countplot(x=col, data=df)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=65)
        chart.set_title(f'{col} Categorical Distribution')
    plt.show()


def show_hist_by_target(
        df: pd.DataFrame,
        columns: List[str]) -> None:
    for col in columns:
        fig, ax = plt.subplots(figsize=(12, 4),
                               nrows=1, ncols=2, squeeze=False)
        sns.violinplot(x='income', y=col, data=df, ax=ax[0][0])
        sns.distplot(df[col], ax=ax[0][1])

    plt.show()


def precision_recall_curve_plot(
        y_test: np.ndarray,
        pred_proba_c1: np.ndarray) -> None:
    precision, recalls, thresholds =\
        precision_recall_curve(y_test, pred_proba_c1)

    plt.figure(figsize=(8, 6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precision[0:threshold_boundary],
             linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary], label='recall')

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))

    plt.xlabel('Threshold value')
    plt.ylabel('Precision and Recall value')
    plt.legend()
    plt.grid()
    plt.show()
