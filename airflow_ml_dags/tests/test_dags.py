import pytest
from airflow.models import DagBag


@pytest.fixture()
def dagbag():
    return DagBag(dag_folder="dags/", include_examples=False)


def test_dagbag_import(dagbag):
    assert not dagbag.import_errors


def test_download_data_dag_import(dagbag):
    assert dagbag.dags is not None
    assert "download_data" in dagbag.dags
    dag = dagbag.dags["download_data"]
    assert dag.tasks is not None
    assert len(dag.tasks) == 3


def test_download_data_dag_structure(dagbag):
    dag = dagbag.dags["download_data"]
    structure = {
        "start-download": ["docker-download"],
        "docker-download": ["end-download"],
        "end-download": []
    }
    for name, task in dag.task_dict.items():
        assert task.downstream_task_ids == set(structure[name])


def test_train_weekly_dag_import(dagbag):
    assert dagbag.dags is not None
    assert "train_weekly" in dagbag.dags
    dag = dagbag.dags["train_weekly"]
    assert dag.tasks is not None
    assert len(dag.tasks) == 8


def test_train_weekly_dag_structure(dagbag):
    dag = dagbag.dags["train_weekly"]
    structure = {
        "start-train": ["wait-data", "wait-target"],
        "wait-data": ["docker-preprocess"],
        "wait-target": ["docker-preprocess"],
        "docker-preprocess": ["docker-split"],
        "docker-split": ["docker-train"],
        "docker-train": ["docker-validate"],
        "docker-validate": ["end-train"],
        "end-train": [],
    }
    for name, task in dag.task_dict.items():
        assert task.downstream_task_ids == set(structure[name])


def test_predict_daily_dag_import(dagbag):
    assert dagbag.dags is not None
    assert "predict_daily" in dagbag.dags
    dag = dagbag.dags["predict_daily"]
    assert dag.tasks is not None
    assert len(dag.tasks) == 6


def test_predict_daily_dag_structure(dagbag):
    dag = dagbag.dags["predict_daily"]
    structure = {
        "start-predict": ["wait-data", "wait-model"],
        "wait-data": ["docker-preprocess"],
        "wait-model": ["docker-preprocess"],
        "docker-preprocess": ["docker-predict"],
        "docker-predict": ["end-predict"],
        "end-predict": [],
    }
    for name, task in dag.task_dict.items():
        assert task.downstream_task_ids == set(structure[name])
