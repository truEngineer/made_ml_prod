import os
import logging

import click
import pandas as pd
from sklearn.preprocessing import StandardScaler


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--mode")
def preprocess(input_dir: str, output_dir: str, mode: str):
    logger = logging.getLogger("preprocess")
    logger.info("#### Data preprocessing.")
    logger.info(f"mode: {mode}.")

    if not os.path.exists(input_dir):
        raise ValueError(f"{input_dir} directory not found.")
    data_path = os.path.join(input_dir, "data.csv")
    if not os.path.exists(data_path):
        raise ValueError(f"{data_path} file not found.")

    data = pd.read_csv(data_path)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    data.to_numpy()[:, :] = scaled_data

    if mode == "train":
        target_path = os.path.join(input_dir, "target.csv")
        if not os.path.exists(target_path):
            raise ValueError(f"{target_path} file not found.")
        data["target"] = pd.read_csv(target_path).to_numpy()

    os.makedirs(output_dir, exist_ok=True)
    output_data_path = os.path.join(output_dir, "data.csv")
    data.to_csv(output_data_path, index=False)

    logger.info("Data preprocessed.")


if __name__ == "__main__":
    preprocess()
