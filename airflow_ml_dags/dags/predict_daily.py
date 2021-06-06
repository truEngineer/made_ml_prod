from datetime import timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

from directories import (
    VOLUME_PATH, LAST_MODEL_DIR, PREDICTIONS_DIR,
    RAW_DATA_DIR, PROCESSED_DATA_DIR, START_DATE,
    PROD_DATE,
)


default_args = {
    "owner": "airflow",
    "email": ["admin@example.com"],
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
        "predict_daily",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=START_DATE,
) as dag:
    start_task = DummyOperator(task_id="start-predict")

    wait_data = FileSensor(
        task_id="wait-data",
        filepath="./data/raw/{{ ds }}/data.csv",
        poke_interval=10,
        retries=100,
    )

    wait_model = FileSensor(
        task_id="wait-model",
        filepath=f"./data/models/{PROD_DATE}/model.pkl",
        poke_interval=10,
        retries=100,
    )

    preprocess_cmd = (
        f" --input-dir {RAW_DATA_DIR}"
        f" --output-dir {PROCESSED_DATA_DIR}"
        f" --mode eval"
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        task_id="docker-preprocess",
        command=preprocess_cmd,
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[f"{VOLUME_PATH}:/data"],
    )

    predict_cmd = (
        f" --input-dir {PROCESSED_DATA_DIR}"
        f" --model-dir {LAST_MODEL_DIR}"
        f" --output-dir {PREDICTIONS_DIR}"
    )

    predict = DockerOperator(
        image="airflow-predict",
        task_id="docker-predict",
        command=predict_cmd,
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[f"{VOLUME_PATH}:/data"],
    )

    end_task = DummyOperator(task_id="end-predict")

    start_task >> [wait_data, wait_model] >> preprocess >> predict >> end_task
