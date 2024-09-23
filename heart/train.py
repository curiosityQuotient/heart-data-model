"""
Script to train a sklearn GradientBoostingClassifier on heart data.
"""

import platform
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import mlflow
import mlflow.sklearn

print("Versions:")
print("  MLflow Version:", mlflow.__version__)
print("  Sklearn Version:", sklearn.__version__)
print("  MLflow Tracking URI:", mlflow.get_tracking_uri())
print("  Python Version:", platform.python_version())
print("  Operating System:", platform.system() + " - " + platform.release())
print("  Platform:", platform.platform())

client = mlflow.MlflowClient()

colLabel = "target"


class Trainer:
    def __init__(self, experiment_name, data_path):
        self.experiment_name = experiment_name
        self.data_path = data_path
        self.X_train, self.X_test, self.y_train, self.y_test = self.build_data(
            data_path
        )

        if self.experiment_name:
            mlflow.set_experiment(experiment_name)

    def build_data(self, data_path):
        data = pd.read_csv(data_path)
        # check for negative values (not expected from variables)
        data[data < 0] = pd.NA
        # check for NaNs, report and drop
        orig_rows = data.shape[0]
        nan_count = data.isna().sum().sum()
        if nan_count > 0:
            data = data.dropna()
            dropped_rows = orig_rows - data.shape[0]
            print(f"{nan_count} NaNs found, {dropped_rows} associated rows dropped.")
        train, test = train_test_split(data, test_size=0.30, random_state=42)

        # Model output is target, binary category zero or 1
        X_train = train.drop([colLabel], axis=1)
        X_test = test.drop([colLabel], axis=1)
        y_train = train[[colLabel]]
        y_test = test[[colLabel]]

        return X_train, X_test, y_train, y_test

    def train(self, n_estimators, max_depth):
        mlflow.sklearn.autolog()
        model = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth
        )
        print("Model:\n ", model)
        model.fit(self.X_train, self.y_train)
        model.score(self.X_test, self.y_test)


def main(experiment_name, data_path, n_estimators, max_depth):
    print("Options:")
    for k, v in locals().items():
        print(f"  {k}: {v}")
    print("Processed Options:")
    trainer = Trainer(experiment_name, data_path)
    trainer.train(n_estimators, max_depth)


if __name__ == "__main__":
    exp_name = "GBC experiment"
    path = "./data/heart_data.csv"
    n_est = 50
    max_depth = 2
    main(exp_name, path, n_est, max_depth)
