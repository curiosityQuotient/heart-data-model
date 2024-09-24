import pytest
import pandas as pd
from ..predict import Predictor
import os


def test_predict():
    print(os.getcwd())
    col_heads = [
        "age",
        "sex",
        "chest pain type",
        "resting blood pressure",
        "chol",
        "fasting blood sugar",
        "resting ECG",
        "max heart rate",
        "exang",
        "oldpeak",
        "slope",
        "number vessels flourosopy",
        "thal",
    ]
    test_row = [37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2]
    test_data = pd.DataFrame(data=[test_row], columns=col_heads)
    mdl_path = "./mlruns/493198603292191274/90a28e8d530d4a93abb89ec2eae9cdcb/artifacts/model/model.pkl"  # Mock model with a fake predict method

    # Create an instance of Predictor with a fake model path
    predictor = Predictor(mdl_path)

    # Run the predict method
    predictor.predict(test_data)
    # Check if the predictions are correct
    assert predictor.predictions == [1]  # Expected output
