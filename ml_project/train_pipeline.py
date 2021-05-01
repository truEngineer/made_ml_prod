import sys
import json
import logging
from typing import Tuple

import click

from src.data import (
    read_data, split_train_val_data,
)
from src.entities import (
    TrainPipelineParams,
    read_train_pipeline_params,
)
from src.features import make_features
from src.features.build_features import (
    build_transformer, extract_target,
)
from src.models import (
    train_model, predict_model,
    evaluate_model, dump_model,
)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(train_pipeline_params: TrainPipelineParams) -> Tuple[str, dict]:
    data = read_data(train_pipeline_params.input_data_path)
    logger.info(f"Start train pipeline with params: {train_pipeline_params}")
    logger.info(f"data.shape: {data.shape}")

    train_df, val_df = split_train_val_data(
        data, train_pipeline_params.split_params,
    )
    logger.info(f"train_df.shape: {train_df.shape}")
    logger.info(f"val_df.shape: {val_df.shape}")

    transformer = build_transformer(train_pipeline_params.feature_params)
    transformer.fit(train_df)

    train_features = make_features(transformer, train_df)
    train_target = extract_target(train_df, train_pipeline_params.feature_params)
    logger.info(f"train_features.shape: {train_features.shape}")

    val_features = make_features(transformer, val_df)
    val_target = extract_target(val_df, train_pipeline_params.feature_params)
    logger.info(f"val_features.shape: {val_features.shape}")

    model = train_model(
        train_features,
        train_target,
        train_pipeline_params.model_params,
    )

    preds = predict_model(model, val_features)
    metrics = evaluate_model(preds, val_target)

    # dump metrics
    with open(train_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics: {metrics}")

    # dump model
    try:
        path_to_model = dump_model(model, train_pipeline_params.output_model_path)
        logger.info(f"Model serialization: {train_pipeline_params.output_model_path}")
    except FileNotFoundError:
        path_to_model = None
        logger.warning(f"Model serialization ERROR: {train_pipeline_params.output_model_path}")

    return path_to_model, metrics


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_train_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
