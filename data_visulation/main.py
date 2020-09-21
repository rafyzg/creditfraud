import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


class DataVisualiztion:
    """
    DataVisualiztion class for comparing feature importance
    and grasping a better understanding of the data.

    :param path: The path to the dataset
    :type conn_id: str
    """

    def __init__(self, path: str = "../creditfraud.csv") -> None:
        self.path = path
        self.df = pd.read_csv(self.path)

    def explore_time(self) -> None:
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 4))
        bins = 50

        ax1.hist(self.df.Time[self.df.Class == 1], bins=bins)
        ax1.set_title("Fraud")

        ax2.hist(self.df.Time[self.df.Class == 0], bins=bins)
        ax2.set_title("Normal")

        plt.xlabel("Time (in Seconds)")
        plt.ylabel("Number of Transactions")
        plt.show()

    def explore_amount(self) -> None:
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

        ax1.scatter(
            self.df.Time[self.df.Class == 1], self.df.Amount[self.df.Class == 1]
        )
        ax1.set_title("Fraud")

        ax2.scatter(
            self.df.Time[self.df.Class == 0], self.df.Amount[self.df.Class == 0]
        )
        ax2.set_title("Normal")

        plt.xlabel("Time (in Seconds)")
        plt.ylabel("Amount")
        plt.show()

    def explore_features(self) -> None:

        v_features = self.df.ix[:, 1:29].columns
        plt.figure(figsize=(12, 28 * 4))
        gs = gridspec.GridSpec(28, 1)

        for i, cn in enumerate(self.df[v_features]):
            ax = plt.subplot(gs[i])
            sns.distplot(self.df[cn][self.df.Class == 1], bins=50)
            sns.distplot(self.df[cn][self.df.Class == 0], bins=50)
            ax.set_xlabel("")
            ax.set_title("histogram of feature: " + str(cn))

        plt.show()


if __name__ == "__main__":
    obj = DataVisualiztion("../creditcard.csv")
    obj.explore_amount()
