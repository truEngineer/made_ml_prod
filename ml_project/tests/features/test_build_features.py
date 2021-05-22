import pytest
import pandas as pd
from numpy.testing import assert_allclose

from src.entities import FeatureParams
from src.data import read_data
from src.features.build_features import (
    make_features, extract_target,
    build_transformer, CustomScaler,
)


def test_custom_scaler(feature_params: FeatureParams, data_path: str):
    data = read_data(data_path)
    num_cols = feature_params.numerical_features
    data = data[num_cols]
    transformer = CustomScaler()
    transformer.fit(data)
    transformed_data = transformer.transform(data).to_numpy()
    assert transformed_data.mean() == pytest.approx(0, 0.1)
    assert transformed_data.std() == pytest.approx(1, 0.1)


def test_make_features(feature_params: FeatureParams, data_path: str):
    data = read_data(data_path)
    transformer = build_transformer(feature_params).fit(data)
    features = make_features(transformer, data)
    assert not pd.isnull(features).any().any()


def test_extract_target(feature_params: FeatureParams, data_path: str):
    data = read_data(data_path)
    target_from_data = data[feature_params.target_col].to_numpy()
    target_extracted = extract_target(data, feature_params).to_numpy()
    assert_allclose(target_from_data, target_extracted)
