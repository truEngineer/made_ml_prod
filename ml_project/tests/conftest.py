import os
from typing import List, Tuple

import pytest
import numpy as np
import pandas as pd

from src.entities import (
    SplitParams, FeatureParams, ModelParams,
)
from src.data import read_data
from src.models.model_fit_predict import (
    train_model, Classifier,
)
from src.features.build_features import (
    make_features, extract_target, build_transformer,
)


@pytest.fixture(scope="session")
def data_path(gen_data: pd.DataFrame) -> str:
    path = os.path.join(os.path.dirname(__file__), "fake_heart.csv")
    gen_data.to_csv(path)
    return path


@pytest.fixture(scope="session")
def data(data_path: str) -> pd.DataFrame:
    return read_data(data_path)


@pytest.fixture(scope="session")
def split_params() -> SplitParams:
    return SplitParams(val_size=0.2, random_state=42)


@pytest.fixture(scope="session")
def target_col() -> str:
    return "target"


@pytest.fixture(scope="session")
def age_threshold_val() -> int:
    return np.random.randint(40, 70)


@pytest.fixture(scope="session")
def numerical_features() -> List[str]:
    return ["age", "trestbps", "chol", "thalach", "oldpeak"]


@pytest.fixture(scope="session")
def categorical_features() -> List[str]:
    return ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


@pytest.fixture(scope="session")
def feature_params(
        categorical_features: List[str],
        numerical_features: List[str],
        target_col: str,
        age_threshold_val: int,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
        age_threshold=age_threshold_val,
    )
    return params


@pytest.fixture(scope="session")
def feature_params_no_thresh(
        categorical_features: List[str],
        numerical_features: List[str],
        target_col: str,
        age_threshold_val: int,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
        age_threshold=None,
    )
    return params


@pytest.fixture(scope="session")
def features_and_target(
        data_path: str, feature_params: FeatureParams,
) -> Tuple[pd.DataFrame, pd.Series]:
    data = read_data(data_path)
    transformer = build_transformer(feature_params).fit(data)
    features = make_features(transformer, data)
    target = extract_target(data, feature_params)
    return features, target


@pytest.fixture(scope="session")
def model(
        features_and_target: Tuple[pd.DataFrame, pd.Series],
) -> Classifier:
    features, target = features_and_target
    return train_model(features, target, ModelParams())


@pytest.fixture(scope="session")
def gen_data():
    size = 100

    heart_data = pd.DataFrame()
    # categorical features
    heart_data["sex"] = np.random.binomial(n=1, p=0.68, size=size)
    heart_data["cp"] = np.random.randint(low=0, high=4, size=size)
    heart_data["fbs"] = np.random.binomial(n=1, p=0.15, size=size)
    heart_data["restecg"] = np.random.randint(low=0, high=3, size=size)
    heart_data["exang"] = np.random.binomial(n=1, p=0.33, size=size)
    heart_data["slope"] = np.random.randint(low=0, high=3, size=size)
    heart_data["ca"] = np.random.randint(low=0, high=5, size=size)
    heart_data["thal"] = np.random.randint(low=0, high=4, size=size)
    # numeric features
    heart_data["age"] = np.random.normal(loc=54.37, scale=9.08, size=size).astype(int)
    heart_data["trestbps"] = np.random.normal(loc=131.62, scale=17.54, size=size).astype(int)
    heart_data["chol"] = np.random.normal(loc=246.26, scale=51.83, size=size).astype(int)
    heart_data["thalach"] = np.random.normal(loc=149.65, scale=22.91, size=size).astype(int)
    heart_data["oldpeak"] = np.random.exponential(scale=1.16, size=size).astype(int)
    # target column
    heart_data["target"] = np.random.binomial(n=1, p=0.54, size=size)

    return heart_data
