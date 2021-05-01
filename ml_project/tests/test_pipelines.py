from py._path.local import LocalPath

import pandas as pd

from train_pipeline import train_pipeline
from eval_pipeline import eval_pipeline

from src.entities import (
    SplitParams, ModelParams, FeatureParams,
    TrainPipelineParams, EvalPipelineParams,
)


def test_train_pipeline(
        tmpdir: LocalPath,
        gen_data: pd.DataFrame,
        feature_params: FeatureParams,
):
    input_data_path = tmpdir.join("tmp_data.csv")
    gen_data.to_csv(input_data_path)
    expected_model_path = tmpdir.join("model.pkl")

    train_pipeline_params = TrainPipelineParams(
        input_data_path=input_data_path,
        output_model_path=expected_model_path,
        metric_path=tmpdir.join("metrics.json"),
        split_params=SplitParams(),
        feature_params=feature_params,
        model_params=ModelParams()
    )

    real_model_path, metrics = train_pipeline(train_pipeline_params)
    assert real_model_path == expected_model_path
    assert all(score in ["f1_score", "acc_score"] for score in metrics.keys())


def test_eval_pipeline(
        tmpdir: LocalPath, gen_data: pd.DataFrame, feature_params_no_thresh: FeatureParams,
):
    input_data_path = tmpdir.join("tmp_data.csv")
    gen_data.to_csv(input_data_path)
    expected_preds_path = tmpdir.join("heart_preds.csv")

    eval_pipeline_params = EvalPipelineParams(
        input_data_path=input_data_path,
        input_model_path="models/model.pkl",
        output_data_path=expected_preds_path,
        feature_params=feature_params_no_thresh,
    )

    real_preds_path, preds = eval_pipeline(eval_pipeline_params)
    assert real_preds_path == expected_preds_path
    assert preds.sum() <= preds.shape[0]
