# dslr

*This project has been created as part of the 42 curriculum by ssian and axlee.*

## Description
`dslr` (Data Science x Logistic Regression) is an introductory machine learning project from the 42 school curriculum. The primary goal is to learn how to read, analyze, and visualize a dataset, and subsequently train a machine learning model to classify data points without relying on heavy external machine learning libraries like `scikit-learn` for the core logic.

The project revolves around the "Harry Potter" universe, where the objective is to classify students into their respective Hogwarts Houses (Gryffindor, Hufflepuff, Ravenclaw, or Slytherin) based on their scores in various magical classes.

The project is divided into three main parts:
1. **Data Description**: Recreating the core functionality of pandas' `describe()` method to analyze the dataset's statistical properties.
2. **Data Visualization**: Generating specific plots to analyze the distribution and correlation of features.
3. **Logistic Regression**: Implementing a One-vs-All logistic regression model using gradient descent to classify the students.

### Benchmarks & Goals
The main goal is to successfully classify the students into the four houses with a high degree of accuracy.
For project validation, the trained model must achieve an accuracy of at least **98%** on the evaluation dataset.

## Technical Choices & Features

* **Data Analysis (`describe.py`)**: A custom implementation that calculates key statistical metrics (count, mean, standard deviation, minimum, maximum, and percentiles) for all numerical features in the dataset, handling `NaN` values gracefully.
* **Feature Visualization**: 
  * `histogram.py`: Identifies the course with the most homogeneous score distribution across all houses.
  * `scatter_plot.py`: Identifies the two features that are the most similar/correlated.
  * `pair_plot.py`: Analyzes all features against each other in a scatter matrix to determine which subsets of features are best suited for logistic regression.
* **Logistic Regression Algorithm**: 
  * Implemented using **Gradient Descent** to minimize the cost/loss function.
  * Uses the **Sigmoid function** to map predictions to probabilities between 0 and 1.
  * Extends binary classification to multi-class classification using the **One-vs-All** (or One-vs-Rest) strategy, training four separate classifiers (one for each house).
* **Data Standardization**: Features are scaled/normalized before training to ensure gradient descent converges efficiently and avoids overflow issues.

## Instructions

### Setup and Installation
To set up the environment and install necessary dependencies (like `pandas`, `numpy`, and `matplotlib`), you can use the provided `requirements.txt` or initialization script:

```bash
source ./init.sh
```

### Cleanup
To clean up the virtual environment, output files and the cache files:
```bash
source ./remove.sh
```

### Execution

**1. Data Description**
To view the statistical summary of the dataset:
```bash
python describe.py data/dataset_train.csv
```

**2. Data Visualization**
To generate the visualizations used for feature selection and analysis:
```bash
python histogram.py ./data/dataset_train.csv
python scatter_plot.py ./data/dataset_train.csv
python pair_plot.py ./data/dataset_train.csv
```

**3. Model Training**
Train the logistic regression model on the training dataset. This will calculate the necessary weights and save them to a file (e.g., `weights.csv`):
```bash
python logreg_train.py ./data/dataset_train.csv
```

**4. Prediction**
Use the trained weights to predict the houses of the students in the test dataset. This generates a `houses.csv` file with the final predictions:
```bash
python logreg_predict.py ./data/dataset_test.csv weights.csv
```

### Error Handling
The scripts are built to handle various edge cases gracefully, such as:
* Missing or empty files.
* Invalid data formatting or non-numeric values in numeric columns.
* Missing values (`NaN` dropping or imputation).
* Incorrect number of command-line arguments.

## Resources

* [42 Curriculum - dslr](https://cdn.intra.42.fr/pdf/pdf/135086/en.subject.pdf)
* [Wikipedia: Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
* [Coursera / Andrew Ng: Machine Learning (Gradient Descent & Logistic Regression)](https://www.coursera.org/learn/machine-learning)
* [Pandas Documentation](https://pandas.pydata.org/docs/)
* [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
