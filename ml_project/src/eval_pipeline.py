import sys
import logging
from typing import Optional, Tuple

import click
import numpy as np

from src.data import read_data
from src.entities import EvalPipelineParams, read_eval_pipeline_params
from src.features import make_features
from src.features.build_features import build_transformer
from src.models import predict_model, load_model


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def eval_pipeline(
        eval_pipeline_params: EvalPipelineParams
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    data = read_data(eval_pipeline_params.input_data_path)
    logger.info(f"Start eval pipeline with params: {eval_pipeline_params}")
    logger.info(f"data.shape: {data.shape}")

    transformer = build_transformer(eval_pipeline_params.feature_params)
    transformer.fit(data)

    features = make_features(transformer, data)
    logger.info(f"features.shape: {features.shape}")

    # load model
    try:
        model = load_model(eval_pipeline_params.input_model_path)
        logger.info(f"Model loaded: {eval_pipeline_params.input_model_path}")
    except FileNotFoundError:
        logger.warning(f"Model loading ERROR: {eval_pipeline_params.input_model_path}")
        return None, None

    preds = predict_model(model, features)
    data["preds"] = preds
    path_to_preds = eval_pipeline_params.output_data_path
    data.to_csv(path_to_preds)
    logger.info(f"Predictions saved: {path_to_preds}")

    return path_to_preds, preds


@click.command(name="eval_pipeline")
@click.argument("config_path")
def eval_pipeline_command(config_path: str):
    params = read_eval_pipeline_params(config_path)
    eval_pipeline(params)


if __name__ == "__main__":
    eval_pipeline_command()
