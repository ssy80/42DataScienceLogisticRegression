import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
from utils import load
from pandas.plotting import scatter_matrix


def plot_pairplot(df: pd.DataFrame):
    """
    Display a pair plot of the most relevant features for predicting Hogwarts House.

    This function performs basic feature selection before visualization:

    1. Drops the "Index" column if present.
    2. Selects only numerical features.
    3. Removes highly correlated features (absolute correlation > 0.9)
       to reduce multicollinearity.
    4. Computes between-house variation by:
       - Calculating mean score per house
       - Computing the standard deviation of those means
    5. Selects the top 5 features with the largest between-house variation,
       as these are most discriminative for classification.
    6. Displays a seaborn pair plot of the selected features, colored
       by "Hogwarts House".
    """

    # Drop index column
    df = df.drop(columns=["Index"])

    df_numeric = df.select_dtypes(include="number")

    # Get correlation matrix
    corr_matrix = df_numeric.corr().abs()

    # Keep only upper triangle
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Drop features with correlation > 0.9
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    df_reduced = df_numeric.drop(columns=to_drop)

    # Get between house variation
    mean_house_scores = df.groupby("Hogwarts House").mean(numeric_only=True)
    between_house_std = mean_house_scores.std()

    # Keep features with larger between-house variation
    important_features = between_house_std.sort_values(ascending=False).head(5).index
    #print("Top separating features:", important_features)

    cols = list(important_features) + ["Hogwarts House"]

    g = sns.pairplot(
        df[cols],
        hue="Hogwarts House",
        diag_kind="hist"
    )

    g._legend.set_title("Hogwarts House")
    g.fig.set_size_inches(12, 16)

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
        plot_pairplot(data_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()