from datetime import timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.providers.docker.operators.docker import DockerOperator

from directories import (
    VOLUME_PATH, START_DATE, MODELS_DIR,
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
)


default_args = {
    "owner": "airflow",
    "email": ["admin@example.com"],
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
        "train_weekly",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=START_DATE,
) as dag:
    start_task = DummyOperator(task_id="start-train")

    wait_data = FileSensor(
        task_id="wait-data",
        poke_interval=10,
        timeout=6000,
        retries=100,
        filepath="./data/raw/{{ ds }}/data.csv",
    )

    wait_target = FileSensor(
        task_id="wait-target",
        poke_interval=10,
        timeout=6000,
        retries=100,
        filepath="./data/raw/{{ ds }}/target.csv",
    )

    preprocess_cmd = (
        f" --input-dir {RAW_DATA_DIR}"
        f" --output-dir {PROCESSED_DATA_DIR}"
        f" --mode train"
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        task_id="docker-preprocess",
        command=preprocess_cmd,
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[f"{VOLUME_PATH}:/data"],
    )

    split_cmd = (
        f" --work-dir {PROCESSED_DATA_DIR}"
        f" --train-size 0.8"
    )

    split = DockerOperator(
        image="airflow-split",
        task_id="docker-split",
        command=split_cmd,
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[f"{VOLUME_PATH}:/data"],
    )

    train_cmd = (
        f" --input-dir {PROCESSED_DATA_DIR}"
        f" --output-dir {MODELS_DIR}"
    )

    train = DockerOperator(
        image="airflow-train",
        task_id="docker-train",
        command=train_cmd,
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[f"{VOLUME_PATH}:/data"],
    )

    validate_cmd = (
        f" --input-dir {PROCESSED_DATA_DIR}"
        f" --model-dir {MODELS_DIR}"
    )

    validate = DockerOperator(
        image="airflow-validate",
        task_id="docker-validate",
        command=validate_cmd,
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[f"{VOLUME_PATH}:/data"],
    )

    end_task = DummyOperator(task_id="end-train")

    start_task >> [wait_data, wait_target] >> preprocess >> split >> train >> validate >> end_task
