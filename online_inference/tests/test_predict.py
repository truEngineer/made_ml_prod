import pandas as pd
from fastapi.testclient import TestClient

from src.app import app


def test_app_main(model_path, config_path, data_path):
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "It is entry point of our predictor" in response.text


def test_prediction_request(model_path, config_path, data_path):
    with TestClient(app) as client:
        data_df = pd.read_csv(data_path).drop("target", axis=1)
        data_df["id"] = data_df.index + 1
        request_data = data_df.values.tolist()[:50]
        request_features = data_df.columns.tolist()
        response = client.get(
            "/predict/", json={"data": request_data, "features": request_features},
        )
        assert response.status_code == 200
        assert sum([x["disease"] for x in response.json()]) <= len(request_data)


def test_prediction_incorrect_feature_count(model_path, config_path, data_path):
    with TestClient(app) as client:
        data_df = pd.read_csv(data_path)
        request_data = data_df.values.tolist()
        request_features = data_df.columns.tolist()
        response = client.get(
            "/predict/", json={"data": request_data, "features": request_features},
        )
        assert response.status_code == 400
        assert "Incorrect feature count or order!" in response.text


def test_prediction_incorrect_feature_order(model_path, config_path, data_path):
    with TestClient(app) as client:
        data_df = pd.read_csv(data_path)
        request_data = data_df.values.tolist()
        request_features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                            "thalach", "exang", "oldpeak", "slope", "ca", "id", "thal"]
        response = client.get(
            "/predict/", json={"data": request_data, "features": request_features},
        )
        assert response.status_code == 400
        assert "Incorrect feature count or order!" in response.text
