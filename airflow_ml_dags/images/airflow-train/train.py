import os
import json
import pickle
import logging

import click
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV  # StratifiedKFold
# from sklearn.ensemble import RandomForestClassifier


@click.command("train")
@click.option("--input-dir")
@click.option("--output-dir")
def train(input_dir: str, output_dir: str):
    logger = logging.getLogger("train")
    logger.info("#### Train model.")

    if not os.path.exists(input_dir):
        raise ValueError(f"{input_dir} directory not found.")

    train_path = os.path.join(input_dir, "train.csv")
    if not os.path.exists(train_path):
        raise ValueError(f"{train_path} file not found.")

    logger.info(f"Load train data: {train_path}.")
    train_data = pd.read_csv(train_path)
    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]

    logger.info("Train model.")
    #model = RandomForestClassifier(random_state=42)
    #param_grid = {"n_estimators": [10, 30, 50], "max_depth": [1, 5, 10]}
    #cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    #grid = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1)
    #grid.fit(X_train, y_train)
    #logger.info(f"Best accuracy: {grid.best_score_:.2f}")
    #logger.info(f"Best params: {grid.best_params_}")
    param_grid = {"C": np.logspace(-3, 3, 10)}
    logreg = LogisticRegression(multi_class="multinomial")  # cross-entropy loss
    logreg_cv = GridSearchCV(logreg, param_grid, cv=5)  # stratified
    logreg_cv.fit(X_train, y_train)

    logger.info(f"Best accuracy: {logreg_cv.best_score_:.2f}")

    logger.info("Save model and metrics.")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(logreg_cv.best_estimator_, f)  # grid

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({"accuracy": logreg_cv.best_score_}, f)  # grid

    logger.info("Model ready.")


if __name__ == "__main__":
    train()
