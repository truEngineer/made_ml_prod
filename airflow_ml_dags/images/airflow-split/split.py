import os
import logging

import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command("split")
@click.option("--work-dir")
@click.option("--train-size", type=float)
def split(work_dir: str, train_size: float):
    logger = logging.getLogger("split")
    logger.info("#### Split data.")

    if not os.path.exists(work_dir):
        raise ValueError(f"{work_dir} directory not found.")
    data_path = os.path.join(work_dir, "data.csv")
    if not os.path.exists(data_path):
        raise ValueError(f"{data_path} file not found.")

    data = pd.read_csv(data_path)
    train_data, val_data = train_test_split(
        data, train_size=train_size, stratify=data["target"],
    )
    train_data.to_csv(os.path.join(work_dir, "train.csv"), index=False)
    val_data.to_csv(os.path.join(work_dir, "val.csv"), index=False)

    logger.info("Data splitted.")


if __name__ == "__main__":
    split()
