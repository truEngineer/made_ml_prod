import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    Binarizer, OneHotEncoder, StandardScaler,
)

from src.entities import FeatureParams


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return num_pipeline


def build_age_threshold_pipeline(age_threshold: int) -> Pipeline:
    age_threshold_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="median")),
            ("binarize", Binarizer(threshold=age_threshold)),
        ]
    )
    return age_threshold_pipeline


def transform_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(df))


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer_config = [
        (
            "categorical_pipeline",
            build_categorical_pipeline(),
            params.categorical_features,
        ),
        (
            "numerical_pipeline",
            build_numerical_pipeline(),
            params.numerical_features,
        ),
    ]
    if params.age_threshold is not None:
        transformer_config.append(
            (
                "age_threshold_pipeline",
                build_age_threshold_pipeline(params.age_threshold),
                ["age"],
            )
        )

    transformer = ColumnTransformer(transformer_config)

    return transformer
