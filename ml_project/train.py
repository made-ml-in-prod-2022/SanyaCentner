import click

from enities import read_training_params
from models import run_train_pipeline


def train_pipeline(config_path: str):
    training_pipeline_params = read_training_params(config_path)
    run_train_pipeline(training_pipeline_params)


@click.command(name="train_model")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    train_pipeline(config_path)


if __name__ == "__main__":
    train_pipeline_command()
