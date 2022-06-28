import os
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split


def split(args):
    df = pd.read_csv(args.data_file)
    data = df.drop(columns=['target'], axis=1)
    target = df['target']

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.15, random_state=42)

    with open(args.otput_file + "/x_train.csv", "w+") as f:
        f.write(x_train.to_csv(index=False))

    with open(args.otput_file + "/y_train.csv", "w+") as f:
        f.write(y_train.to_csv(index=False))

    with open(args.otput_file + "/x_test.csv", "w+") as f:
        f.write(x_test.to_csv(index=False))

    with open(args.otput_file + "/y_test.csv", "w+") as f:
        f.write(y_test.to_csv(index=False))


def createparser():
    """read argument"""
    parser_args = argparse.ArgumentParser()
    parser_args.add_argument(
        "--data", type=str, default="data.csv", help="data file path", dest="data_file"
    )
    parser_args.add_argument(
        "--split", type=str, help="split data file path", dest="otput_file"
    )
    return parser_args


if __name__ == '__main__':
    parser = createparser()
    namespace = parser.parse_args()
    split(namespace)
