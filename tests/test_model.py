"""
MLOps Assignment No. 7: CI/CD Pipeline Using GitHub Actions
Module: tests/test_model.py
Description: Unit test for evaluating model accuracy using pytest.
"""

import sys
import os

# Add project root directory to sys.path so 'src.model' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import train_model


def test_model_accuracy():
    """
    Unit test to verify that the trained model achieves an accuracy >= 0.80.
    """
    accuracy = train_model()
    assert accuracy >= 0.80
