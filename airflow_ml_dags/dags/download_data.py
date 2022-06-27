import os
from datetime import timedelta
from datetime import datetime
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(seconds=5),
}

with DAG(
        "data_generate",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(8)
) as dag:

    start = DummyOperator(task_id="start")

    data_generate = DockerOperator(
        image="airflow-model-download",
        command="python download.py --data /data/raw/{{ ds }}/data.csv --target /data/raw/{{ ds }}/target.csv",
        network_mode="bridge",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/Users/aleksandr/Documents/SanyaCentner/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    start >> data_generate
