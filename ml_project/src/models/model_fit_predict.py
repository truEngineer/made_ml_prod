import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

from src.entities.model_params import ModelParams

Classifier = Union[LogisticRegression, RandomForestClassifier]


def train_model(
        features: pd.DataFrame, target: pd.Series, model_params: ModelParams,
) -> Classifier:
    if model_params.model_type == "LogisticRegression":
        model = LogisticRegression(
            C=model_params.inv_regularization_strength,
            solver="liblinear",
            intercept_scaling=model_params.intercept_scaling,
        )
    elif model_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=model_params.n_estimators,
            random_state=model_params.random_state,
        )
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(model: Classifier, features: pd.DataFrame) -> np.ndarray:
    return model.predict(features)


def evaluate_model(preds: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "f1_score": f1_score(target, preds),
        "acc_score": accuracy_score(target, preds),
    }


def dump_model(model: LogisticRegression, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def load_model(input: str) -> Classifier:
    with open(input, "rb") as f:
        model = pickle.load(f)
    return model
