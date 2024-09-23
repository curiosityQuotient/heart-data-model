"""
Script to produce a prediction from a sklearn GradientBoostingClassifier on heart data.
"""

import platform
import pandas as pd
import sklearn
import pickle

print("Versions:")
print("  Sklearn Version:", sklearn.__version__)
print("  Python Version:", platform.python_version())
print("  Operating System:", platform.system() + " - " + platform.release())
print("  Platform:", platform.platform())


class Predictor:
    def __init__(self, mdl_pkl):
        self.model = mdl_pkl

    def predict(self, data):
        with open(self.model, "rb") as pickle_file:
            model = pickle.load(pickle_file)
        self.predictions = model.predict(data)


def main(mdl_pkl, data):
    print("Options:")
    for k, v in locals().items():
        print(f"  {k}: {v}")
    predictor = Predictor(mdl_pkl)
    predictor.predict(data)
    prediction = predictor.predictions
    print(prediction)


if __name__ == "__main__":
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
    mdl_path = "./mlruns/493198603292191274/90a28e8d530d4a93abb89ec2eae9cdcb/artifacts/model/model.pkl"
    main(mdl_path, test_data)
