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
    "retries": 1,
    "retry_delay": timedelta(seconds=5),
}


def _wait_for_split_data():
    return os.path.exists("/opt/airflow/data/split/{{ ds }}/x_train.csv")\
            and os.path.exists("/opt/airflow/data/split/{{ ds }}/x_test.csv")\
            and os.path.exists("/opt/airflow/data/split/{{ ds }}/y_train.csv")\
            and os.path.exists("/opt/airflow/data/split/{{ ds }}/y_test.csv")


with DAG(
        "train_model",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(8),
) as dag:

    start = DummyOperator(task_id="start")

    data_preprocess = DockerOperator(
        image="airflow-model-preprocess",
        command="python preprocess.py --data /data/raw/{{ ds }} --processed /data/processed/{{ ds }}/train_data.csv",
        network_mode="bridge",
        task_id="docker-aiflow-model-preprocess",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/Users/aleksandr/Documents/SanyaCentner/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    data_split = DockerOperator(
        image="airflow-model-split",
        command="python split.py --data /data/processed/{{ ds }}/train_data.csv --split /data/split/{{ ds }}",
        network_mode="bridge",
        task_id="docker-aiflow-model-split",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/Users/aleksandr/Documents/SanyaCentner/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    train = DockerOperator(
        image="airflow-model-train",
        command="python train.py --split /data/split/{{ ds }} --model /data/models/{{ ds }}",
        network_mode="bridge",
        task_id="docker-aiflow-model-train",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/Users/aleksandr/Documents/SanyaCentner/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    val = DockerOperator(
        image="airflow-model-validate",
        command="python validate.py --split /data/split/{{ ds }} --model /data/models/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-model-validate",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[
            Mount(source="/Users/aleksandr/Documents/SanyaCentner/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    start >> data_preprocess >> data_split >> train >> val