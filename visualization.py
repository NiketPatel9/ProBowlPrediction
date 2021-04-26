"""
This module is responsible for for creating various interesting visualizations on the data, both before
pre-processing and after pre-processing. It will also visualize SMOTE-generated data.
"""

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd


def histograms(passing, title="raw"):
    """
    This function will plot histograms of each column and will color Pro Bowlers versus non Pro Bowlers.

    :param passing: A pandas dataframe whose columns will be plotted.
    :param title: A string to indicate the type of column and whether it is preprocessed
    :return: void
    """

    for col in passing.columns[1:len(passing.columns)]:
        sns.displot(passing, x=col, hue=passing.columns[0], discrete=True)
        plt.savefig(f"visualizations/{col}_histogram_{title}.png")


def boxplots(passing):
    """
    This function creates a series of boxplots in order to visualize outliers.

    :param passing: a pandas dataframe whose columns will be used for boxplot visualizations.

    :return: void
    """
    for col in passing.columns[1:len(passing.columns)]:
       sns.boxplot(y=passing[col])
       plt.savefig(f"visualizations/{col}_boxplot.png")
       plt.clf()


def pairplotting(passing):
    """
    This function creates a pairplot of the passing data to show correlation between variables.

    :param passing: a pandas dataframe whose correlation between variables will be analyzed.
    :return: void
    """
    sns.pairplot(passing, hue="pro_bowl")
    plt.savefig(f"visualizations/pairplot.png")
    plt.clf()


def distributions(passing):
    """
    This function will fit a distribution curve for each column.

    :param passing: A pandas dataframe whose columns will be fitted for distributions.
    :return: void
    """
    for col in passing.columns[1:len(passing.columns)]:
        sns.displot(passing, x=col, hue=passing.columns[0], kind="kde", multiple="stack")
        plt.savefig(f"visualizations/{col}_distribution.png")
        plt.clf()


def pca_plot(df, target,  title):
    """
    This function performs PCA on the given dataframe as well as plots the results.

    :param df: A dataframe that will be used for the PCA fit.
    :param target: A dataframe representing the outcome variable to the corresponding row in the features dataframe
    :param title: A string to identify the saved plot
    :return: void
    """
    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(df)

    principalDf = pd.DataFrame(data = principalComponents, columns=['Principal Component 1',
                                                                    'Principal Component 2'])

    finalDf = pd.concat([principalDf, target], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    targets = [0, 1]
    colors = ['r', 'g']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['pro_bowl'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'Principal Component 1'],
                   finalDf.loc[indicesToKeep, 'Principal Component 2'],
                   c=color,
                   s=50)
    ax.legend(targets)
    ax.grid()

    plt.savefig(f"visualizations/pca_{title}.png")
    plt.clf()
