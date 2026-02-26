import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load(path: str) -> pd.DataFrame:
    """
    Load a csv file path into a dataframe,
    return the dataframe if success, else return None.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None
    return df


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

    # Drop index column
    df = df.drop(columns=["Index"])

    df_numeric = df.select_dtypes(include="number")

    corr_matrix = df_numeric.corr().abs()

    # Keep only upper triangle (exclude diagonal)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Get best corr pair, upper.stack() convert to multiindex[row_label, column_label]
    max_corr = upper.stack().idxmax()     # get index(r, c) with highest corr value

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
 
        dataset_train_filepath = "./data/dataset_train.csv"

        data_df = load(dataset_train_filepath)

        plot_scatterplot(data_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
