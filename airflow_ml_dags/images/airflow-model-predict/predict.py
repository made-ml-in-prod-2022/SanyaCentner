import os
import pickle
import pandas as pd
import numpy as np
import click


@click.command("predict")
@click.option('--data-dir')
@click.option('--model-path')
@click.option('--output-dir')
def predict(data_dir, output_dir):
    data = pd.read_csv(os.path.join(data_dir, "data.csv"))
    with open(model_path, mode='rb') as f:
        model = pickle.load(f)

    predicts = model.predict(data)

    os.makedirs(output_dir, exist_ok=True)
    predicts_df = pd.DataFrame(predicts)

    with open(os.path.join(output_dir, "predictions.csv"), "wb") as f:
        predicts_df.to_csv(f, index=False)


if __name__ == '__main__':
    predict()
