# Airflow

## Start Airflow:

```bash
user$ docker compose up --build
```

## Fernet keygen:

```bash
user$ docker run airflow-docker python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)"
```

Add generated key as environment variable `AIRFLOW__CORE__FERNET_KEY` in `docker-compose.yml`.

## Tests:

```bash 
user$ docker compose up -d --build
user$ docker ps
CONTAINER ID   IMAGE          COMMAND                  CREATED         STATUS         PORTS                                       NAMES
4cadbd2bada6   c6c5c9f74212   "/usr/bin/dumb-init …"   3 minutes ago   Up 2 minutes   8080/tcp                                    airflow_ml_dags_scheduler_1
7661c60ec3aa   c6c5c9f74212   "/usr/bin/dumb-init …"   3 minutes ago   Up 2 minutes   0.0.0.0:8080->8080/tcp, :::8080->8080/tcp   airflow_ml_dags_webserver_1
f1c707a74650   0d05bac255ba   "docker-entrypoint.s…"   3 minutes ago   Up 2 minutes   0.0.0.0:5432->5432/tcp, :::5432->5432/tcp   airflow_ml_dags_postgres_1
user$ docker exec -it 4cadbd2bada6 bash
root@4cadbd2bada6:/opt/airflow# pip install pytest
root@4cadbd2bada6:/opt/airflow# pytest -v .
```

Самооценка (42 балла):

1. Поднимите airflow локально, используя docker compose (можно использовать из [примера](https://github.com/made-ml-in-prod-2021/airflow-examples/)) ✅

2. (5 баллов) Реализуйте dag, который генерирует данные для обучения модели (генерируйте данные, можете использовать как генератор синтетики из первой дз, так и что-то из датасетов sklearn), вам важно проэмулировать ситуации постоянно поступающих данных 
   - записывайте данные в `/data/raw/{{ ds }}/data.csv`, `/data/raw/{{ ds }}/target.csv` ✅

3. (10 баллов) Реализуйте dag, который обучает модель еженедельно, используя данные за текущий день. В вашем пайплайне должно быть как минимум 4 стадии, но дайте волю своей фантазии=)️ ✅

   - подготовить данные для обучения (например, считать из `/data/raw/{{ ds }}` и положить `/data/processed/{{ ds }}/train_data.csv`)
   - разделить их на train/val
   - обучить модель на train (сохранить в `/data/models/{{ ds }}`
   - провалидировать модель на val (сохранить метрики к модельке)

4. (5 баллов) Реализуйте dag, который использует модель ежедневно ✅

   - принимает на вход данные из пункта 1 (data.csv)
   - считывает путь до модельки из airflow variables (идея в том, что когда нам нравится другая модель и мы хотим ее на прод)
   - делает предсказание и записывает их в `/data/predictions/{{ ds }}/predictions.csv`
   - (3 доп. балла) реализуйте сенсоры на то, что данные готовы для дагов тренировки и обучения ✅

5. Вы можете выбрать 2 пути для выполнения ДЗ

   - (0 баллов) поставить все необходимые пакеты в образ с airflow и использовать bash operator, python operator
   - использовать DockerOperator, тогда выполнение каждой из тасок должно запускаться в собственном контейнере
     - (5 баллов) 1 из дагов реализован с помощью DockerOperator ❌
     - (10 баллов) все даги реализованы только с помощью DockerOperator ([пример](https://github.com/made-ml-in-prod-2021/airflow-examples/blob/main/dags/11_docker.py)) ✅  
     
   По технике, вы можете использовать такую же структуру как в примере, пакуя в разные докеры скрипты, можете использовать общий докер с вашим пакетом, но с разными точками входа для разных тасок.

   Прикольно, если вы покажете, что для разных тасок можно использовать разный набор зависимостей. В [этом](https://github.com/made-ml-in-prod-2021/airflow-examples/blob/main/dags/11_docker.py#L27) месте пробрасывается путь с хостовой машины, используйте здесь путь типа `/tmp` или считывайте из переменных окружения.

6. (5 баллов) [Протестируйте](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html) ваши даги ✅

7. (5 доп. баллов) В docker compose так же настройте поднятие mlflow и запишите туда параметры обучения, метрики и артефакт (модель) ❌

8. (5 доп. баллов) Вместо пути в airflow variables используйте апи Mlflow Model Registry. Даг для инференса подхватывает последнюю продакшен модель. ❌

9. (3 доп. балла) [Настройте](https://www.astronomer.io/guides/error-notifications-in-airflow) alert в случае падения дага ✅

10. (1 балл)️ Самооценка ✅