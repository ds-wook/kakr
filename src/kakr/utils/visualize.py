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
        chart.title(f'{col} Categorical Distribution')
    plt.show()


def show_hist_by_target(
        df: pd.DataFrame,
        columns: List[str]) -> None:
    for col in columns:
        plt.figure(figsize=(12, 4))
        sns.violinplot(x='income', y=col, data=df)
        sns.distplot(df[col])
    plt.show()
