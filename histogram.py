import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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

    #Drop index column
    df = df.drop(columns=["Index"])

    #Mean score per house
    mean_house_scores = df.groupby("Hogwarts House").mean(numeric_only=True)
    
    #Standard deviation across houses
    std_house_values = mean_house_scores.std()
    print(f"Lowest standard deviation course: {std_house_values.idxmin()}")

    #Plot histogram
    course = "Care of Magical Creatures"

    # Define bin edges with width = 1
    min_score = int(np.floor(df[course].min()))
    max_score = int(np.ceil(df[course].max()))
    bins = np.arange(min_score, max_score + 1, 1)

    plt.figure()

    sns.histplot(
        data=df,
        x=course,
        hue="Hogwarts House",
        bins=bins,
        multiple="layer",   # overlay houses
        stat="count",       # y-axis = number of students
        alpha=0.5
    )

    plt.xlabel("Score")
    plt.ylabel("Number of Students")
    plt.title("Care of Magical Creatures Score Distribution by House")

    plt.tight_layout()
    plt.show()


def main():
    """main()"""

    try:
 
        dataset_train_filepath = "./data/dataset_train.csv"

        data_df = load(dataset_train_filepath)

        plot_histogram(data_df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()