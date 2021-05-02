# ML Project

## Installation:  

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt


## Dataset

[Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci). 

`heart.csv` file in `data/raw` directory.


## User Guide

EDA report generation (`reports` folder):

    python generate_report.py

Run tests:

    pytest -v  tests/

Run train/eval pipeline:

    python train_pipeline {train_config_path}
    python eval_pipeline {eval_config_path}


## Project Organization


    ├── configs             <- Configuration files.
    │
    ├── data
    │   ├── preds           <- Model predictions (heart_preds.csv).
    │   └── raw             <- The original, immutable data dump (heart.csv).
    │
    ├── models              <- Trained and serialized models.
    │
    ├── notebooks           <- Jupyter notebooks (EDA).
    │
    ├── reports             <- Generated EDA report files.
    │
    ├── src                 <- Source code for use in this project.
    │   │
    │   ├── entities        <- Configuration dataclasses for type checking.
    │   │
    │   ├── data            <- Code to download or generate data.
    │   │
    │   ├── features        <- Code to turn raw data into features for modeling.
    │   │
    │   ├── models          <- Code to train models and then use trained models to make predictions.
    │
    ├── tests               <- Unit tests for project modules and e2e tests.
    │
    ├── eval_pipeline.py    <- Eval pipeline CLI.
    │
    ├── generate_report.py  <- Generate report from data/raw/heart.csv.
    │
    ├── README.md           <- The top-level README for developers using this project.
    │
    ├── requirements.txt    <- The requirements file for reproducing the analysis environment.
    │
    └── train_pipeline.py   <- Train pipeline CLI.
 
 
Самооценка (32 балла):

+ Назовите ветку homework1 (1 балл) 

+ Положите код в папку ml_project

+ В описании к пулл-реквесту описаны основные "архитектурные" и тактические решения, которые сделаны в вашей работе. В общем, описание что именно вы сделали и для чего, чтобы вашим ревьюерам было легче понять ваш код. (2 балла)

+ Выполнение EDA, закоммитьте ноутбук в папку с ноутбуками (2 балла)

+ Вы так же можете построить в ноутбуке прототип (если это вписывается в ваш тиль работы). Можете использовать не ноутбук, а скрипт, который сгенерит отчет, закоммитьте и скрипт и отчет (1 балл)

+ Проект имеет модульную структуру (2 балла)

+ Использованы логгеры (2 балла)

+ Написаны тесты на отдельные модули и на прогон всего пайплайна (3 балла)

+ Для тестов генерируются синтетические данные, приближенные к реальным (3 балла)

+ Обучение модели конфигурируется с помощью конфигов в json или yaml, закоммитьте как минимум 2 корректные конфигурации, с помощью которых можно обучить модель (разные модели, стратегии split, preprocessing) (3 балла)

+ Используются датаклассы для сущностей из конфига, а не голые dict (3 балла)

+ Используйте кастомный трансформер (написанный своими руками) и протестируйте его (3 балла)

+ Обучите модель, запишите в readme как это предполагается (3 балла)

+ Напишите функцию predict, которая примет на вход артефакт/ы от обучения, тестовую выборку (без меток) и запишет предикт, напишите в readme как это сделать (3 балла)

+ Проведите самооценку, опишите, в какое колво баллов по вашему мнению стоит оценить вашу работу и почему (1 доп. балл)
