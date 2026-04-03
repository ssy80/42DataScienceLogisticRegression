import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
from utils import load


def plot_histogram(df: pd.DataFrame):
    """
    Display a histogram of the score distribution for the most homogeneous course.

    This function:
    1. Drops the "Index" column if present.
    2. Computes the mean score per Hogwarts House for each numeric course.
    3. Calculates the standard deviation of house means to determine
       between-house variation.
    4. Identifies the course with the lowest between-house standard deviation
       (i.e., the most homogeneous course across houses).
    5. Plots a histogram of student scores for "Care of Magical Creatures"
       grouped by Hogwarts House.
    """

    df = df.drop(columns=["Index"])
    mean_house_scores = df.groupby("Hogwarts House").mean(numeric_only=True)
    
    std_house_values = mean_house_scores.std()
    print(f"Lowest standard deviation course: {std_house_values.idxmin()}")

    course =  std_house_values.idxmin()
    min_score = int(np.floor(df[course].min()))
    max_score = int(np.ceil(df[course].max()))
    bins = np.arange(min_score, max_score + 1, 1)

    plt.figure()

    sns.histplot(
        data=df,
        x=course,
        hue="Hogwarts House",
        bins=bins,
        multiple="layer",
        stat="count",
        alpha=0.5
    )

    plt.xlabel("Score")
    plt.ylabel("Number of Students")
    plt.title("Care of Magical Creatures Score Distribution by House")

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
        plot_histogram(data_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()