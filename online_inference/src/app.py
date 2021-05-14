import os
import pickle
import logging
from typing import List, Optional

import uvicorn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.responses import PlainTextResponse

from src.features.build_features import (
    build_transformer,
    transform_features,
)
from .entities import (
    AppRequest, AppResponse,
    read_eval_pipeline_params,
)


logger = logging.getLogger(__name__)


model: Optional[LogisticRegression] = None
transformer: Optional[ColumnTransformer] = None


def load_object(path: str) -> LogisticRegression:
    with open(path, "rb") as f:
        return pickle.load(f)


def make_predict(
        data: List,
        features: List[str],
        model: LogisticRegression,
        transformer: ColumnTransformer,
) -> List[AppResponse]:
    data = pd.DataFrame(data, columns=features)
    ids = data["id"]
    features_df = data.drop(["id"], axis=1)

    transformed_features_df = transform_features(transformer, features_df)
    preds = model.predict(transformed_features_df)

    return [
        AppResponse(id=id, disease=int(disease)) for id, disease in zip(ids, preds)
    ]


app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


@app.get("/")
def main():
    return "It is entry point of our predictor"


@app.on_event("startup")
def load_model():
    logger.info(f"Loading model...")
    global model
    model_path = os.getenv("PATH_TO_MODEL")
    if model_path is None:
        err = "PATH_TO_MODEL is None"
        logger.error(err)
        raise RuntimeError(err)
    model = load_object(model_path)
    logger.info(f"Model is ready...")


@app.on_event("startup")
def build_transformer_1():
    logger.info(f"Building transformer...")
    global transformer
    config_path = os.getenv("PATH_TO_CONFIG")
    if config_path is None:
        err = "PATH_TO_CONFIG is None"
        logger.error(err)
        raise RuntimeError(err)
    data_path = os.getenv("PATH_TO_DATA")
    if data_path is None:
        err = "PATH_TO_DATA is None"
        logger.error(err)
        raise RuntimeError(err)
    params = read_eval_pipeline_params(config_path)
    transformer = build_transformer(params.feature_params)
    transformer.fit(pd.read_csv(data_path).drop(["target"], axis=1))
    logger.info(f"Transformer is ready...")


@app.get("/predict/", response_model=List[AppResponse])
def predict(request: AppRequest):
    return make_predict(request.data, request.features, model, transformer)


if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
