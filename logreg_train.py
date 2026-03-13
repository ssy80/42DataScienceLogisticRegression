import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import load


def save_data(weights_df: pd.DataFrame):
    """Save weights to file"""
    
    filename = "weights.csv"

    weights_df.to_csv(filename, index=False)
    print(f"Weights successfully saved to {filename}")


def sigmoid(z):
    """
    Compute the sigmoid (logistic) function.
    Formula:
        sigmoid(z) = 1 / (1 + exp(-z))
    """
    return 1 / (1 + np.exp(-z))


def compute_cost(train_X, train_y, theta):
    """
    Compute the logistic loss (cost) for given features, labels, and weights.
    Formula:
        cost = -(1/m) * [y * log(prediction) + (1 - y) * log(1 - prediction)]
    """

    m  =  len(train_X)
    prediction = sigmoid(train_X @ theta)

    # Needed to add epislon to avoid log(0) which is undefined
    epislon = 1e-15

    cost = -(1/m) * np.sum(train_y * np.log(prediction + epislon) + (1 - train_y) * np.log(1 - prediction + epislon))
    return cost


def train_logistic_regression_gradient_descent(train_X: pd.DataFrame, train_y: pd.DataFrame, learning_rate: float, iterations: int, house: str):
    """
    Train a binary logistic regression model using gradient descent
    in a one-vs-rest (OvR) classification setting.

    For the given `house`, the target labels are converted into a binary
    vector where samples belonging to `house` are labeled as 1 and all
    other samples are labeled as 0. The model parameters (weights) are
    optimized by minimizing the logistic loss using gradient descent.
    """

    # Initialize weights (theta) to zeros
    theta = np.zeros(train_X.shape[1])

    m = len(train_X) # number of rows or observations

    for i in range(iterations):

        # Calculate the prediction error
        prediction = sigmoid(train_X @ theta) #0-1 probality
        error = prediction - train_y

        # Calculate the gradient (partial derivative)
        gradient = (1/m) * (train_X.T @ error)

        # Update weights for next iteration training
        theta -= (learning_rate * gradient)

        if i % 100 == 0:
            cost = compute_cost(train_X, train_y, theta)
            print(f"House: {house:10} | Iteration {i:4} | Cost: {cost:.4f}")
        
    final_cost = compute_cost(train_X, train_y, theta)
    print(f"House: {house:10} | Iteration {iterations} | Cost: {final_cost:.4f}")
        
    return theta


def train_logistic_regression_multi(train_X: pd.DataFrame, train_y: pd.DataFrame, learning_rate: float, iterations: int):
    """
    Train a multi-class logistic regression model using the
    one-vs-rest (OvR) strategy and gradient descent.

    For each unique class label in `train_y`, a separate binary
    logistic regression model is trained to distinguish that class
    from all other classes. A bias term is added to the feature
    matrix before training. The learned weight vectors are stored
    and returned in a dictionary keyed by class label.
    """
    weights = {}

    houses = sorted(train_y.unique())

    # Add Bias column (column of 1s) to X
    train_X = np.c_[np.ones((train_X.shape[0], 1)), train_X]

    for house in houses:
        
        y_binary = np.where(train_y == house, 1, 0) #train each class(1) vs others(0)

        w = train_logistic_regression_gradient_descent(train_X, y_binary, learning_rate, iterations, house)
        weights[house] = w

    return weights


def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the dataset by selecting features, handling missing values,
    and applying feature scaling.

    This function separates the target variable ("Hogwarts House") from the
    input features, removes non-numeric or unused columns, imputes missing
    values using the median strategy, and standardizes the features to have
    zero mean and unit variance.
    """

    y = df["Hogwarts House"]

    to_drop_cols = ["Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"]
    X = df.drop(columns=to_drop_cols)

    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    #Standard Scaling
    standard_scaler  = StandardScaler()
    X = standard_scaler.fit_transform(X)

    return (X, y)


def predict(test_X: pd.DataFrame, weights_df: pd.DataFrame):
    """
    Predict class labels for a multi-class logistic regression model
    trained using a one-vs-rest (OvR) strategy.

    This function computes class probabilities for each sample by applying
    the sigmoid function to the linear scores obtained from the feature
    matrix and the learned weight vectors. A bias term is added to the input
    features before prediction. The predicted class for each sample is the
    one with the highest probability.
    """

    houses = weights_df.columns.tolist()
    theta = weights_df.values
    
    test_X = np.c_[np.ones((test_X.shape[0], 1)), test_X] # insert bias

    z = test_X @ theta
    probabilities = sigmoid(z)

    prediction_indices = np.argmax(probabilities, axis=1)
    
    final_predictions = [houses[i] for i in prediction_indices]

    return final_predictions


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

        train_df, test_df = train_test_split(
                        data_df,
                        test_size=0.2,
                        random_state=42,
                        stratify=data_df["Hogwarts House"]
                    )

        train_X, train_y = preprocess_data(train_df)
        test_X, test_y = preprocess_data(test_df)
        
        weights = train_logistic_regression_multi(train_X, train_y, 0.01, 1000)

        weights_df = pd.DataFrame(weights)
        save_data(weights_df)

        # Get predictions for test_X set
        predictions = predict(test_X, weights_df)

        # Calculate accuracy for test_X, test_y
        score = accuracy_score(test_y, predictions)
        print(f"Accuracy: {score * 100:.2f}%")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()