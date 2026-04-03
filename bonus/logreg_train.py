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
import argparse


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
    epislon = 1e-15

    cost = -(1/m) * np.sum(train_y * np.log(prediction + epislon) + (1 - train_y) * np.log(1 - prediction + epislon))
    return cost


def train_logistic_regression_gradient_descent(train_X: np.ndarray, train_y: pd.Series, learning_rate: float, iterations: int, house: str):
    """
    Train a binary logistic regression model using gradient descent
    in a one-vs-rest (OvR) classification setting.

    For the given `house`, the target labels are converted into a binary
    vector where samples belonging to `house` are labeled as 1 and all
    other samples are labeled as 0. The model parameters (weights) are
    optimized by minimizing the logistic loss using gradient descent.
    """

    theta = np.zeros(train_X.shape[1])

    m = len(train_X)

    for i in range(iterations):

        prediction = sigmoid(train_X @ theta)
        error = prediction - train_y

        gradient = (1/m) * (train_X.T @ error)
        theta -= (learning_rate * gradient)

        if i % 100 == 0:
            cost = compute_cost(train_X, train_y, theta)
            print(f"House: {house:10} | Iteration {i:4} | Cost: {cost:.4f}")
        
    final_cost = compute_cost(train_X, train_y, theta)
    print(f"House: {house:10} | Iteration {iterations} | Cost: {final_cost:.4f}")
        
    return theta


def train_logistic_regression_sgd(train_X: np.ndarray, train_y: np.ndarray, learning_rate: float, epochs: int, house: str):
    """
    Train a binary logistic regression model using stochastic gradient descent (SGD).

    This function iteratively updates the weight vector by computing the gradient
    of the logistic loss for each training sample and adjusting the weights in
    the direction that minimizes the cost.

    Runs for a number of epoch and returns the final weight vector 
    Cost is printed every 10 epochs.
    """

    if not isinstance(train_X, np.ndarray) or not isinstance(train_y, np.ndarray):
        raise TypeError("train_X and train_y must be numpy arrays.")
    
    if train_X.size == 0 or train_y.size == 0:
        raise ValueError("Training data cannot be empty.")
        
    if train_X.shape[0] != train_y.shape[0]:
        raise ValueError("The number of samples in train_X and train_y must be equal.")
        
    if learning_rate <= 0 or epochs <= 0:
        raise ValueError("learning_rate and epochs must be strictly positive.")

    m = len(train_X)
    n = train_X.shape[1]
    theta = np.zeros(n)

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = train_X[indices]
        y_shuffled = train_y[indices]

        for i in range(m):
            xi = X_shuffled[i]
            yi = y_shuffled[i]

            prediction = sigmoid(np.dot(xi, theta))
            error = prediction - yi
            gradient = xi * error
            theta -= learning_rate * gradient
        
        if epoch % 10 == 0:
            cost = compute_cost(train_X, train_y, theta)
            print(f"House: {house:10} | Epoch {epoch:4} | Cost: {cost:.4f}")

    return theta

def train_logistic_regression_minibatch(train_X: np.ndarray, train_y: np.ndarray, learning_rate: float, epochs: int, house: str, batch_size=32):
    """
    Train a binary logistic regression model using mini-batch gradient descent.

    This function iteratively updates the weight vector by computing the gradient
    of the logistic loss for mini-batches of training samples and adjusting the weights
    in the direction that minimizes the cost.

    Runs for a number of epoch and returns the final weight vector 
    Cost is printed every 10 epochs.
    """

    if not isinstance(train_X, np.ndarray) or not isinstance(train_y, np.ndarray):
        raise TypeError("train_X and train_y must be numpy arrays.")
    
    if train_X.size == 0 or train_y.size == 0:
        raise ValueError("Training data cannot be empty.")
        
    if train_X.shape[0] != train_y.shape[0]:
        raise ValueError("The number of samples in train_X and train_y must be equal.")
        
    if learning_rate <= 0 or epochs <= 0:
        raise ValueError("learning_rate and epochs must be strictly positive.")
    
    n = train_X.shape[1]
    m = len(train_X)
    theta = np.zeros(n)

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = train_X[indices]
        y_shuffled = train_y[indices]

        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            
            prediction = sigmoid(xi @ theta)
            error = prediction - yi
            gradient = xi.T @ error
            theta -= (learning_rate * gradient)

        if epoch % 100 == 0:
            cost = compute_cost(train_X, train_y, theta)
            print(f"House: {house:10} | Epoch {epoch:4} | Cost: {cost:.4f}")

    return theta

def train_logistic_regression_multi(train_X: np.ndarray, train_y: pd.Series, learning_rate: float, iterations: int, algorithm: str = "batch"):
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

    train_X = np.c_[np.ones((train_X.shape[0], 1)), train_X]

    for house in houses:
        y_binary = np.where(train_y == house, 1, 0)

        if algorithm == "sgd":
            w = train_logistic_regression_sgd(train_X, y_binary, learning_rate, iterations, house)
        elif algorithm == "minibatch": 
            w = train_logistic_regression_minibatch(train_X, y_binary, learning_rate, iterations, house)
        else:
            w = train_logistic_regression_gradient_descent(train_X, y_binary, learning_rate, iterations, house)
        weights[house] = w

    return weights


def preprocess_data(df: pd.DataFrame, imputer=None, standard_scaler=None):
    """
    Preprocess the dataset by selecting features, handling missing values,
    and applying feature scaling.

    This function separates the target variable ("Hogwarts House") from the
    input features, removes non-numeric or unused columns, imputes missing
    values, and standardizes the features to have
    zero mean and unit variance.
    """

    y = df["Hogwarts House"]

    to_drop_cols = ["Index",
                    "First Name",
                    "Last Name",
                    "Birthday",
                    "Best Hand",
                    "Hogwarts House",
                    "Care of Magical Creatures",
                    "Potions",
                    "Divination",
                    "Herbology",
                    "History of Magic",
                    #"Arithmancy",
                    #"Astronomy",
                    #"Defense Against the Dark Arts",
                    #"Muggle Studies",
                    #"Ancient Runes",
                    #"Transfiguration",
                    #"Charms",
                    #"Flying"
                    ]
    X = df.drop(columns=to_drop_cols)

    if imputer is None:
        imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    else:
        X = pd.DataFrame(imputer.transform(X), columns=X.columns)

    if standard_scaler is None:
        standard_scaler  = StandardScaler()
        X = pd.DataFrame(standard_scaler.fit_transform(X), columns=X.columns)
    else:
        X = pd.DataFrame(standard_scaler.transform(X), columns=X.columns)

    return (X, y, imputer, standard_scaler)


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
    
    test_X = np.c_[np.ones((test_X.shape[0], 1)), test_X]

    z = test_X @ theta
    probabilities = sigmoid(z)

    prediction_indices = np.argmax(probabilities, axis=1)
    
    final_predictions = [houses[i] for i in prediction_indices]

    return final_predictions


def main():
    """main()"""

    try:
        parser = argparse.ArgumentParser(description="Train logistic regression models.")
        parser.add_argument("dataset",
                            help="Path to the training dataset CSV file")
        parser.add_argument("--algo",
                            choices=["batch", "sgd", "minibatch"],
                            default="batch", 
                            help="Optimization algorithm to use (default: batch)")
        args = parser.parse_args()

        data_df = load(args.dataset)
        
        if data_df is None:
            print("Error: failed to load data")
            return

        if args.algo == "sgd":
            print("Training using Stochastic Gradient Descent (SGD)...")
            lr = 0.01
            epochs = 100
        elif args.algo == "minibatch":
            print("Training using Mini-Batch Gradient Descent...")
            lr = 0.05
            epochs = 500
        else:
            print("Training using Batch Gradient Descent...")
            lr = 0.1
            epochs = 10000

        train_df, test_df = train_test_split(
                        data_df,
                        test_size=0.2,
                        random_state=42,
                        stratify=data_df["Hogwarts House"]
                    )

        train_X, train_y, fitted_imputer, fitted_scaler = preprocess_data(train_df)
        test_X, test_y, _, _ = preprocess_data(test_df, fitted_imputer, fitted_scaler)
        
        weights = train_logistic_regression_multi(train_X, train_y, lr, epochs, algorithm=args.algo)

        weights_df = pd.DataFrame(weights)
        save_data(weights_df)

        predictions = predict(test_X, weights_df)
        score = accuracy_score(test_y, predictions)
        print(f"Accuracy: {score * 100:.2f}%")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
