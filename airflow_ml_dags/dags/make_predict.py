import os
from datetime import timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from docker.types import Mount

default_args = {
    "owner": "airflow",
    "email": ["sashaabramov1998@gmail.com"],
    'email_on_failure': True,
    "retries": 1,
    "retry_delay": timedelta(seconds=5),
}

with DAG(
        dag_id="make_predict",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(1),
) as dag:
    start = DummyOperator(task_id="start")

    predict = DockerOperator(
        image="airflow-model-predict",
        command="python predict.py --data /data/raw/{{ ds }}/data.csv --model /data/models/{{ ds }}/model.pkl --predict /data/predictions/{{ ds }}/predictions.csv",
        network_mode="bridge",
        task_id="docker-airflow-model-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/Users/aleksandr/Documents/SanyaCentner/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    start >> predict