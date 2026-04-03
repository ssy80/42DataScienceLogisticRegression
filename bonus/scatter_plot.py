import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from utils import load


def plot_scatterplot(df: pd.DataFrame):
    """
    Plot a scatter plot of the two most strongly correlated numeric features.

    This function identifies the pair of numerical features with the highest
    absolute Pearson correlation coefficient and visualizes their relationship
    using a scatter plot.

    The procedure is as follows:
    1. Drops the "Index" column if present.
    2. Selects only numerical columns.
    3. Computes the absolute correlation matrix.
    4. Considers only unique feature pairs (upper triangle, excluding diagonal).
    5. Identifies the pair with the highest correlation.
    6. Displays a scatter plot of those two features.
    """

    df = df.drop(columns=["Index"])
    df_numeric = df.select_dtypes(include="number")

    corr_matrix = df_numeric.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    max_corr = upper.stack().idxmax()

    plt.figure()
    plt.scatter(df_numeric[max_corr[0]], df_numeric[max_corr[1]])
    plt.xlabel(max_corr[0])
    plt.ylabel(max_corr[1])
    plt.title(f"Scatter Plot: {max_corr[0]} vs {max_corr[1]}")
    plt.tight_layout()
    plt.show()


def main():
    """main()"""

    try:
        if len(sys.argv) != 2:
            print("Error: the arguments are bad")
            return

        dataset_train_filepath = str(sys.argv[1])
        data_df = load(dataset_train_filepath)

        if data_df is None:
            print("Error: failed to load data from dataframe. Check loading from train_filepath")
            return
        plot_scatterplot(data_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
