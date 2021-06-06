import os
import logging

import click
import numpy as np
from sklearn.datasets import load_iris


@click.command("download")
@click.option("--output-dir")
def download(output_dir: str):
    logger = logging.getLogger("download")
    logger.info("#### Download data.")

    iris_X, iris_y = load_iris(return_X_y=True, as_frame=True)
    iris_X.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    iris_X.to_numpy()[:, :] += np.random.random(iris_X.shape) * 0.1  # data generation
    os.makedirs(output_dir, exist_ok=True)
    iris_X.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    iris_y.to_csv(os.path.join(output_dir, "target.csv"), index=False)

    logger.info(f"Data downloaded.")


if __name__ == "__main__":
    download()
