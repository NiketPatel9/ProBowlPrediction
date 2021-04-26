"""
This module contains functions that pre-processes dataframes.
"""
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler


def remove_outliers(df):
    """
    Takes in a dataframe and removes rows with outliers in any column, based on Z-scores.

    :param df:  A pandas dataframe whose rows will be removed.
    :return: A Dataframe Object with no huge outliers.
    """
    final_df = df[(np.abs(stats.zscore(df.iloc[:, 1:len(df.columns)])) < 3).all(axis=1)]

    return final_df


def standardize(df):
    """
    Accepts a dataframe and standardizes certain columns in the dataframe

    :param df: The DataFrame to be standardized
    :return: A DataFrame object that has its measurements columns standardized
    """
    standardized_df = df

    for col in standardized_df.columns[1:len(standardized_df.columns)]:
        scale = StandardScaler().fit(standardized_df[[col]])
        standardized_df[col] = scale.transform(standardized_df[[col]])

    return standardized_df