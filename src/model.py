"""
MLOps Assignment No. 7: CI/CD Pipeline Using GitHub Actions
Module: src/model.py
Description: Trains a Logistic Regression model on the Iris dataset.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_model():
    """
    1. Loads the Iris dataset.
    2. Splits the dataset into training and testing sets (test_size=0.2, random_state=42).
    3. Creates a Logistic Regression model with max_iter=200.
    4. Trains the model.
    5. Makes predictions on the testing dataset.
    6. Calculates model accuracy.
    7. Returns the accuracy.
    """
    # 1. Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 2. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Create Logistic Regression model
    model = LogisticRegression(max_iter=200)

    # 4. Train the model
    model.fit(X_train, y_train)

    # 5. Make predictions on testing set
    y_pred = model.predict(X_test)

    # 6. Calculate model accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # 7. Return accuracy
    return accuracy


if __name__ == "__main__":
    print("=" * 50)
    print("Training Logistic Regression Model on Iris Dataset...")
    print("=" * 50)
    accuracy = train_model()
    print("Model Training Completed Successfully!")
    print(f"Model Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("=" * 50)
