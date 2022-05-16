import click
from enities import read_training_params
from models import run_test_pipeline


def predict_model(config_path: str):
    params = read_training_params(config_path)
    run_test_pipeline(params)


@click.command(name="predict_model")
@click.argument("config_path")
def predict_model_command(config_path: str):
    predict_model(config_path)


if __name__ == "__main__":
    predict_model_command()