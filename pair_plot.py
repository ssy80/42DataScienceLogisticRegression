import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
import seaborn as sns
from utils import load


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
    5. Selects the top 4 features with the largest between-house variation,
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
    #print("Dropped due to high correlation:", to_drop)
    #print("Remaining features:", df_reduced.columns)

    # Get between house variation
    mean_house_scores = df.groupby("Hogwarts House").mean(numeric_only=True)
    between_house_std = mean_house_scores.std()

    # Keep features with larger between-house variation
    important_features = between_house_std.sort_values(ascending=False).head(4).index
    #print("Top separating features:", important_features)

    cols = list(important_features) + ["Hogwarts House"]

    g = sns.pairplot(
        df[cols],
        hue="Hogwarts House",     # color by house
        diag_kind="hist"          # histogram on diagonal
    )

    g._legend.set_title("Hogwarts House")
    g.fig.set_size_inches(12, 16)

    plt.show()


def main():
    """main()"""

    try:
 
        dataset_train_filepath = "./data/dataset_train.csv"

        data_df = load(dataset_train_filepath)

        plot_pairplot(data_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()