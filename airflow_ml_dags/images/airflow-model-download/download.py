import click
import os
import pandas as pd
import random

SOURCE_DATA_PATH = 'source_data.csv'


@click.command('download')
@click.argument('output_dir')
def download(output_dir: str):
    df = pd.read_csv(SOURCE_DATA_PATH)
    max_size = df.shape[0]
    n_rows = random.randint(80, max_size)
    df = df.sample(n=n_rows, ignore_index=True)

    data_df = df
    del data_df['condition']
    target_df = df['condition']

    os.makedirs(output_dir, exist_ok=True)
    # data_df.to_csv(os.path.join(output_dir, 'data.csv'), index=False)
    # target_df.to_csv(os.path.join(output_dir, 'target.csv'), index=False)
    with open(os.path.join(output_dir, "data.csv"), "wb") as f:
        data_df.to_csv(f, index=False)
    with open(os.path.join(output_dir, "target.csv"), "wb") as f:
        target_df.to_csv(f, index=False)


if __name__ == '__main__':
    download()
