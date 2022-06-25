import os
import pickle

import numpy as np
import pandas as pd
import click
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


@click.command('train')
@click.option('--input-dir')
@click.option('--output-dir')
def train(input_dir: str, output_dir: str):

    df = pd.read_csv(os.path.join(input_dir, 'train_data.csv'))

    X = df[[col for col in df.columns if col != 'target']]
    y = df['target']

    cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    num_pipeline = Pipeline([
        ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', LogisticRegression())
    ])

    pipeline.fit(X, y)

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'model.pkl'), mode='wb') as f:
        pickle.dump(pipeline, f)


if __name__ == '__main__':
    train()
