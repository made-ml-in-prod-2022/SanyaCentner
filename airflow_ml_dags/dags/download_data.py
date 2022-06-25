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

with DAG(
        dag_id='data_download_dag',
        default_args=default_args,
        schedule_interval='@daily',
        start_date=days_ago(1),
        max_active_runs=1,
) as dag:
    download = DockerOperator(
        image='airflow-download',
        command='/data/raw/{{ ds }}',
        network_mode='bridge',
        task_id='docker-airflow-download',
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=host_data_dir, target='/data', type='bind')]
    )

    download
