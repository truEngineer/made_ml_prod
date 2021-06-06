import os
import pickle
import logging

import pandas as pd
import click


@click.command("predict")
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def predict(input_dir: str, model_dir: str, output_dir: str):
    logger = logging.getLogger("predict")
    logger.info("### Make predictions.")

    if not os.path.exists(input_dir):
        raise ValueError(f"{input_dir} directory not found.")
    if not os.path.exists(model_dir):
        raise ValueError(f"{model_dir} directory not found.")

    data_path = os.path.join(input_dir, "data.csv")
    if not os.path.exists(data_path):
        raise ValueError(f"{data_path} file not found.")

    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise ValueError(f"{model_path} file not found.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded.")

    data = pd.read_csv(data_path)
    preds = model.predict(data)
    preds = pd.DataFrame({"preds": preds})

    os.makedirs(output_dir, exist_ok=True)
    preds.to_csv(os.path.join(output_dir, "predictions.csv"))

    logger.info("Predictions ready.")


if __name__ == "__main__":
    predict()
