# Online Inference

## Docker build

`docker build -t truengineer/online_inference:v1 .`

## Docker push

`docker push truengineer/online_inference:v1`

## Docker pull

`docker pull truengineer/online_inference:v1`

## Docker run

`docker run -p 8000:8000 truengineer/online_inference:v1`

## Docker image size optimization

`FROM python:3.8-slim` (1.3 GB -> 0.6 GB)


Самооценка (22 балла):

- Оберните inference вашей модели в rest-сервис (FastAPI) (3 балла)

- Напишите тест для `predict`  (3 балла)

- Напишите скрипт `make_request`, который будет делать запросы к вашему сервису (2 балла)

- Сделайте валидацию входных данных, возращайте 400, в случае, если валидация не пройдена (3 доп. балла)

- Напишите `dockerfile`, соберите на его основе образ и запустите локально контейнер, внутри которого должен запускаться сервис, написанный в предыдущем пункте, напишите в `readme` корректную команду сборки `docker build` (4 балла)

- Оптимизируйте размер `docker image`, опишите в `readme` что вы предприняли для сокращения размера и каких результатов удалось добиться (3 доп. балла)

- Опубликуйте образ в [dockerhub](https://hub.docker.com/), используя `docker push` (2 балла)

- Напишите в `readme` корректные команды `docker pull/run`, которые должны привести к тому, что локально поднимется на inference ваша модель, убедитесь в работоспособности скрипта `make_request` (1 балл)

- Проведите самооценку (1 доп. балл)
