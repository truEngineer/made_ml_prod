import os
import json
import pickle
import logging

import click
import pandas as pd
from sklearn.metrics import accuracy_score


@click.command("validate")
@click.option("--input-dir")
@click.option("--model-dir")
def validate(input_dir: str, model_dir: str):
    logger = logging.getLogger("validate")
    logger.info("#### Model validation.")

    if not os.path.exists(input_dir):
        raise ValueError(f"{input_dir} directory not found.")
    if not os.path.exists(model_dir):
        raise ValueError(f"{model_dir} directory not found.")

    val_path = os.path.join(input_dir, "val.csv")
    if not os.path.exists(val_path):
        raise ValueError(f"{val_path} file not found.")

    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise ValueError(f"{model_path} file not found.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded.")

    val_data = pd.read_csv(val_path)
    X_val = val_data.drop("target", axis=1)
    y_val = val_data["target"].values
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    logger.info(f"Model accuracy: {accuracy:.2f}.")

    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": accuracy}, f)
    logger.info("Metrics saved.")


if __name__ == "__main__":
    validate()
