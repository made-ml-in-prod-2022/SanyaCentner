import os
import pandas as pd
import click
from sklearn.model_selection import train_test_split


@click.command('split')
@click.option('--input-dir')
@click.option('--output-dir')
def split(input_dir, output_dir):
    data = pd.read_csv(os.path.join(input_dir, 'train_data.csv'))
    train_data, test_data = train_test_split(data, test_size=0.15)
    os.makedirs(output_dir, exist_ok=True)
    train_data.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)


if __name__ == '__main__':
    split()
