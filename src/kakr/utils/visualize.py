import pandas as pd
import matplotlib.pyplot as plt
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
        fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=2, squeeze=False)
        sns.violinplot(x='income', y=col, data=df, ax=ax[0][0])
        sns.distplot(df[col], ax=ax[0][1])

    plt.show()
