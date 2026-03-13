import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load

def plot_boxplot(df: pd.DataFrame):
    """
    Plot a boxplot for the specified column in the dataframe.
    """
    feature = "Astronomy"

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="Hogwarts House",
        y=feature,
        palette="Set2",
        hue="Hogwarts House"
    )

    plt.title(f"Boxplot of {feature} by Hogwarts House")
    plt.xlabel("Hogwarts House")
    plt.ylabel(feature)

    plt.tight_layout()
    plt.show()

def main():
    """
    main()
    """

    try:
        if len(sys.argv) != 2:
            print("Error: the arguments are bad")
            return
        
        dataset_train_filepath = str(sys.argv[1])
        data_df = load(dataset_train_filepath)

        if data_df is None:
            print("Error: failed to load data from dataframe. Check loading from train_filepath")
            return
        plot_boxplot(data_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()