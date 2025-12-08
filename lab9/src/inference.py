import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


def load_model(path: str) -> LogisticRegression:
    with open(path, "rb") as f:
        model = joblib.load(f)
    return model


def predict(model: LogisticRegression, inp: dict[str, float]) -> str:
    x = np.array(list(inp.values())).reshape(1, -1)
    pred = model.predict(x)
    return pred[0]
