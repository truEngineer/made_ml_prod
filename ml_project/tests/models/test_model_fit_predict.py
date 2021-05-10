import os
import pickle
from typing import Tuple
from py._path.local import LocalPath

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.entities import ModelParams
from src.models.model_fit_predict import (
    train_model, predict_model,
    evaluate_model, dump_model,
    Classifier
)


def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model = train_model(
        features=features,
        target=target,
        model_params=ModelParams(),
    )
    assert isinstance(model, LogisticRegression)
    assert model.predict(features).shape[0] == target.shape[0]


def test_predict_model(
        model: Classifier, features_and_target: Tuple[pd.DataFrame, pd.Series],
):
    features, target = features_and_target
    preds = predict_model(model, features)
    assert preds.sum() <= preds.shape[0]


def test_evaluate_model(
        model: Classifier, features_and_target: Tuple[pd.DataFrame, pd.Series],
):
    features, target = features_and_target
    preds = predict_model(model, features)
    metrics = evaluate_model(preds, target)
    assert all(val >= 0 for val in metrics.values())


def test_dump_model(tmpdir: LocalPath):
    expected_output = tmpdir.join("model.pkl")
    model = LogisticRegression()
    real_output = dump_model(model, expected_output)
    assert real_output == expected_output
    assert os.path.exists
    with open(real_output, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, LogisticRegression)
