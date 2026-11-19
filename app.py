"""
MLOps Assignment No. 7: CI/CD Pipeline Using GitHub Actions
Application Entry Point: app.py
A simple command-line ML application demonstrating model execution and reporting.
"""

from src.model import train_model


def main():
    print("=" * 60)
    print("       MLOps CI/CD Demo - Machine Learning Application       ")
    print("=" * 60)
    print("Application Name  : Iris Classification App")
    print("ML Algorithm      : Logistic Regression (max_iter=200)")
    print("Dataset           : Iris Dataset (scikit-learn)")
    print("-" * 60)
    print("Training and evaluating model...")

    accuracy = train_model()

    print(f"Model Accuracy    : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("-" * 60)
    print("Execution Status  : Application executed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
