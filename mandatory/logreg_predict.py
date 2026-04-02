import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from logreg_train import sigmoid, preprocess_data, predict
from utils import load


def save_predictions(predictions: list):
    """
    Save predictions to file
    """
    filename = "houses.csv"
    
    predictions_df = pd.DataFrame(predictions, columns=["Hogwarts House"])
    predictions_df.index.name = "Index"

    predictions_df.to_csv(filename)
    print(f"predictions successfully saved to {filename}")


def main():
    """main()"""

    try:
        if len(sys.argv) != 3:
            print("Error: the arguments are bad")
            return

        data_filepath = str(sys.argv[1])
        weights_filepath = str(sys.argv[2])
        
        data_df = load(data_filepath)
        weights_df = load(weights_filepath)

        if data_df is None or weights_df is None:
            print("Error: failed to load data or weights")
            return

        test_X, _, _, _ = preprocess_data(data_df)

        predictions = predict(test_X, weights_df)
        
        save_predictions(predictions)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
