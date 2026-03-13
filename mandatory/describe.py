import pandas as pd
import numpy as np
import sys
from utils import load


def drop_non_numerical_features(df: pd.DataFrame):
    """
    Return a dataframe with only numerical columns.
    """

    return df.select_dtypes(include=[np.number])


def get_count(series: pd.Series) -> int:
    """
    Count non-null values in a series.
    """
    
    count = 0
    for value in series:
        if (pd.isna(value)):
            continue
        else:
            count += 1
    
    return int(count)


def get_mean(series: pd.Series) -> float:
    """
    Compute mean manually, ignoring null values.
    """

    total = 0.0
    count = 0

    for value in series:
        if pd.isna(value):
            continue
        total += value
        count += 1

    if count == 0:
        return float("nan")

    return float(total / count)


def get_standard_deviation(series: pd.Series) -> float:
    """
    Compute standard deviation manually, ignoring null values.
        1)Compute the mean
        2)Subtract mean from each value
        3)Square the differences
        4)Sum them
        5)Divide by n - 1 (sample)
        6)Take square root
    """

    mean = get_mean(series)
    if pd.isna(mean):
        return float("nan")

    squared_diff_sum = 0.0
    count = 0

    for value in series:
        if pd.isna(value):
            continue
        diff = value - mean
        squared_diff_sum += (diff**2)
        count += 1

    if count < 2:
        return float("nan")

    variance = squared_diff_sum / (count - 1)
    std = float(np.sqrt(variance))
    
    return std


def get_min(series: pd.Series):
    """
    Compute min manually, ignoring null values.
    """

    min_value = None

    for value in series:
        if pd.isna(value):
            continue

        if (min_value is None) or (value < min_value):
            min_value = value

    return min_value


def get_max(series: pd.Series):
    """
    Compute max manually, ignoring null values.
    """

    max_value = None

    for value in series:
        if pd.isna(value):
            continue
        if (max_value is None) or (value > max_value):
            max_value = value

    return max_value


def get_percent(series: pd.Series, percent: float) -> float:
    """
    Compute percentile manually (linear interpolation), ignoring null values.
    """

    values = []
    for value in series:
        if pd.isna(value):
            continue
        values.append(float(value))

    n = len(values)
    if n == 0:
        return float("nan")

    values.sort()

    if percent <= 0:
        return values[0]
    if percent >= 1:
        return values[-1]

    position = (n - 1) * percent
    lower_index = int(np.floor(position))
    upper_index = int(np.ceil(position))

    if lower_index == upper_index:
        return values[lower_index]

    lower_value = values[lower_index]
    upper_value = values[upper_index]
    fraction = position - lower_index

    return lower_value + (upper_value - lower_value) * fraction


def describe(df: pd.DataFrame)-> pd.DataFrame:
    """
    Generate descriptive statistics for numerical columns in a DataFrame.

    This function replicates the core behavior of `pandas.DataFrame.describe()`
    for numeric features. It performs the following steps:

    - Drops the "Index" column if present.
    - Removes non-numerical columns.
    - Computes descriptive statistics for each remaining column:
        - Count (non-null values)
        - Mean
        - Standard deviation (sample standard deviation)
        - Minimum
        - 25th percentile
        - 50th percentile (median)
        - 75th percentile
        - Maximum
    """

    # Drop index column
    df = df.drop(columns=["Index"])

    # Drop non-numerical featutres
    df = drop_non_numerical_features(df)

    describe_dict = {}
    for col_name in df.columns:
        count = get_count(df[col_name])
        mean = get_mean(df[col_name])
        std = get_standard_deviation(df[col_name])
        min_value = get_min(df[col_name])
        max_value = get_max(df[col_name])
        percent_25 = get_percent(df[col_name], 0.25)
        percent_50 = get_percent(df[col_name], 0.5)
        percent_75 = get_percent(df[col_name], 0.75)
        
        feature = [
            count,
            mean,
            std,
            min_value,
            percent_25,
            percent_50,
            percent_75,
            max_value
        ]
        describe_dict[col_name] = feature

    index_labels = [
        "Count",
        "Mean",
        "Std",
        "Min",
        "25%",
        "50%",
        "75%",
        "Max"
    ]

    describe_df = pd.DataFrame(describe_dict, index=index_labels)

    return describe_df
    

def main():
    """main()"""

    try:
        if len(sys.argv) != 2:
            print("Error: the arguments are bad")
            return

        data_filepath = str(sys.argv[1])
        
        data_df = load(data_filepath)
        if data_df is None:
            print("Error: failed to load data")
            return
        
        print("\nOriginal dataframe: \n", data_df.describe())
        describe_df = describe(data_df)
        print("\nManual Describe dataframe: \n", describe_df)


    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
