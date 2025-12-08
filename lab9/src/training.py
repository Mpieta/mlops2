import os
import joblib
import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.utils import Bunch


def load_data() -> Bunch:
    return load_iris()


def train_model() -> LogisticRegression:
    data = load_data()
    X, y = data.data, data.target_names[data.target]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def save_model(model: LogisticRegression, path: str) -> None:
    with open(path, "wb") as f:
        joblib.dump(model, f)


if __name__ == "__main__":
    model = train_model()
    save_model(model, os.path.join("models", "iris.joblib"))
