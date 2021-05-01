import pandas as pd

from src.data.make_dataset import read_data, split_train_val_data
from src.entities import SplitParams


def test_read_data(data_path: str, target_col: str):
    data = read_data(data_path)
    assert len(data) > 10
    assert target_col in data.keys()


def test_split_train_val_data(data: pd.DataFrame, split_params: SplitParams):
    train_data, val_data = split_train_val_data(data, split_params)
    assert len(train_data) > 10
    assert len(val_data) > 10
