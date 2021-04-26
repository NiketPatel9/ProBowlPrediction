"""
This module will be the driver of the entire project. It will call functions that web scrapes
the ProFootballReference online database for Quarterback stats throughout many years of the NFL,
cleans up this data, and saves it to a CSV file. It will also run visualizations on this data before pre-processing,
handle outliers, pre-process the data, handle label imbalance, create and save the models, and then
evaluate the models with multiple metrics.
"""
from retrieve_csv import create_csvs, rename_csv_cols
from preprocess import remove_outliers, standardize
from visualization import histograms, boxplots, pairplotting, distributions, pca_plot
from models import random_forest_experiment, logistic_regression_experiment, neural_network_experiment, \
    k_fold_experiment, improved_random_forest, bagging_experiment
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def main():
    """
    This is the driver function for the entire project, and will call relevant helper functions.

    :return:
    """
    create_csvs()
    rename_csv_cols()

    passing = pd.read_csv("passing.csv")

    # Visualizatiions
    histograms(passing)
    boxplots(passing)
    pairplotting(passing)
    distributions(passing)

    # Preprocessing
    passing = remove_outliers(passing)
    passing = standardize(passing)
    passing = passing.drop("Tie", axis=1)
    histograms(passing, "standardized")

    # Train-test split and label balancing of training set via SMOTE, as well as visualization via PCA
    X = passing.iloc[:, 1:len(passing.columns)]
    y = passing.pop("pro_bowl")

    pca_plot(X, y, "presmote")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)

    full_X = pd.concat([X_train, X_test])
    full_y = pd.concat([y_train, y_test])
    full_y = full_y.reset_index(drop=True)
    pca_plot(full_X, full_y, "smote")

    # Base level classification experiments
    random_forest_experiment(X_train, X_test, y_train, y_test)
    logistic_regression_experiment(X_train, X_test, y_train, y_test)
    neural_network_experiment(X_train, X_test, y_train, y_test)

    # K-fold cross validation
    k_fold_experiment(full_X, full_y)

    # Tuning hyperparameters
    improved_random_forest(X_train, X_test, y_train, y_test)

    # Simple Bagging
    bagging_experiment(full_X, full_y)


if __name__ == "__main__":
    main()
