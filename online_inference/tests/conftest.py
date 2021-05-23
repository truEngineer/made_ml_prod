import os

import pytest


@pytest.fixture(scope="session")
def model_path() -> str:
    os.environ["PATH_TO_MODEL"] = "models/model.pkl"
    return os.getenv("PATH_TO_MODEL")


@pytest.fixture(scope="session")
def config_path() -> str:
    os.environ["PATH_TO_CONFIG"] = "configs/eval_logreg_config.yaml"
    return os.getenv("PATH_TO_CONFIG")


@pytest.fixture(scope="session")
def data_path() -> str:
    os.environ["PATH_TO_DATA"] = "data/heart.csv"
    return os.getenv("PATH_TO_DATA")
