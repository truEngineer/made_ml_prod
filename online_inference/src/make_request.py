import logging

import numpy as np
import pandas as pd
import requests
import click


logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=8000)
@click.option("--data_path", default="data/heart.csv")
def predict(host, port, data_path):
    data = pd.read_csv(data_path).drop("target", axis=1)
    data["id"] = data.index + 1
    request_features = data.columns.tolist()
    for i in range(100):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]

        logger.info(f"Request: {request_data}")
        response = requests.get(
            f"http://{host}:{port}/predict/",
            json={"data": [request_data], "features": request_features},
        )

        logger.info(f"Response code: {response.status_code}, body: {response.json()}")


if __name__ == "__main__":
    predict()
