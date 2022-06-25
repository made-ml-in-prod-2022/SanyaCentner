from datetime import timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount


default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

host_data_dir = Variable.get('data_dir')
mounts = [Mount(source=host_data_dir, target='/data', type='bind')]

with DAG(
        'model_train_dag',
        default_args=default_args,
        schedule_interval='@weekly',
        start_date=days_ago(1),
) as dag:
    preprocess = DockerOperator(
        image='airflow-preprocess',
        command='--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}',
        network_mode='bridge',
        task_id='docker-airflow-preprocess',
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=mounts
    )

    split = DockerOperator(
        image='airflow-split',
        command='--input-dir /data/processed/{{ ds }} --output-dir /data/split/{{ ds }} --val_size 0.33',
        network_mode='bridge',
        task_id='docker-airflow-split',
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=mounts
    )

    train = DockerOperator(
        image='airflow-train',
        command='--input-dir /data/split/{{ ds }} --output-dir /data/models/{{ ds }}',
        network_mode='bridge',
        task_id='docker-airflow-train',
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=mounts
    )

    evaluate = DockerOperator(
        image='airflow-evaluate',
        command='--input-dir /data/split/{{ ds }} --models-dir /data/models/{{ ds }} --output-dir /data/metrics/{{ ds }}',
        network_mode='bridge',
        task_id='docker-airflow-evaluate',
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=mounts
    )

    preprocess >> split >> train >> evaluate
