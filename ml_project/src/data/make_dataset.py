import sys
import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.entities import SplitParams


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def read_data(path: str) -> pd.DataFrame:
    logger.info(f"Reading data from {path}")
    data = pd.read_csv(path)
    logger.info(f"Reading finished")
    return data


def split_train_val_data(
    data: pd.DataFrame, split_params: SplitParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Splitting data")
    train_data, val_data = train_test_split(
        data,
        test_size=split_params.val_size,
        random_state=split_params.random_state,
    )
    logger.info("Splitting finished")
    return train_data, val_data
