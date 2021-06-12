from datetime import timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator

from directories import (
    VOLUME_PATH, RAW_DATA_DIR, START_DATE,
)


default_args = {
    "owner": "airflow",
    "email": ["admin@example.com"],
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
        "download_data",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=START_DATE,
) as dag:
    start_task = DummyOperator(task_id="start-download")

    download_data = DockerOperator(
        image="airflow-download",
        task_id="docker-download",
        command=RAW_DATA_DIR,
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[f"{VOLUME_PATH}:/data"],
    )

    end_task = DummyOperator(task_id="end-download")

    start_task >> download_data >> end_task
