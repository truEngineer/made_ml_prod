from airflow.models import Variable
from airflow.utils.dates import days_ago


START_DATE = days_ago(2)

VOLUME_PATH = Variable.get("volume_path")
PROD_DATE = Variable.get("prod_date")

RAW_DATA_DIR = "/data/raw/{{ ds }}"
PROCESSED_DATA_DIR = "/data/processed/{{ ds }}"
MODELS_DIR = "/data/models/{{ ds }}"
PREDICTIONS_DIR = "/data/predictions/{{ ds }}"

LAST_MODEL_DIR = f"/data/models/{PROD_DATE}"
