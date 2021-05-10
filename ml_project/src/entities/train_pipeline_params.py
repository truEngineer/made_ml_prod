import sys
import logging

import yaml
from dataclasses import dataclass
from marshmallow_dataclass import class_schema

from .split_params import SplitParams
from .feature_params import FeatureParams
from .model_params import ModelParams


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@dataclass()
class TrainPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    split_params: SplitParams
    feature_params: FeatureParams
    model_params: ModelParams


TrainingPipelineParamsSchema = class_schema(TrainPipelineParams)


def read_train_pipeline_params(path: str) -> TrainPipelineParams:
    with open(path, "r") as input_stream:
        config_dict = yaml.safe_load(input_stream)
        schema = TrainingPipelineParamsSchema().load(config_dict)
        logger.info(f"schema: {schema}")
        return schema
