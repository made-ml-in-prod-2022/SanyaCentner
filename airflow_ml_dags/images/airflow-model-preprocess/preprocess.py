import os
import pandas as pd
import click


@click.command('preprocess')
@click.option('--input-dir')
@click.option('--output-dir')
def preprocess(input_dir: str, output_dir):
    data_df = pd.read_csv(os.path.join(input_dir, 'data.csv'))
    target_df = pd.read_csv(os.path.join(input_dir, 'target.csv'))
    data_df['target'] = target_df['condition']

    os.makedirs(output_dir, exist_ok=True)
    data_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)


if __name__ == '__main__':
    preprocess()
