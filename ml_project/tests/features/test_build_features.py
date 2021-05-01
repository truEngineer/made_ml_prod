import pandas as pd
from numpy.testing import assert_allclose

from src.entities import FeatureParams
from src.data import read_data
from src.features.build_features import (
    make_features, extract_target, build_transformer,
)


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
