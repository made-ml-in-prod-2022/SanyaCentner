import os
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


def download(args):
    data, target = make_classification()

    with open(args.data_file, "w+") as f:
        f.write(pd.DataFrame(data, columns=[f"feature_{i}" for i in range(20)]).to_csv(index=False))

    with open(args.target_file, "w+") as f:
        f.write(pd.DataFrame(target, columns=["Target"]).to_csv(index=False))


def createparser():
    """read argument"""
    parser_args = argparse.ArgumentParser()
    parser_args.add_argument(
        "--data", type=str, default="data.csv", help="data file path", dest="data_file"
    )
    parser_args.add_argument(
        "--target", type=str, default="target.csv", help="target file path", dest="target_file"
    )
    return parser_args


if __name__ == '__main__':
    parser = createparser()
    namespace = parser.parse_args()
    download(namespace)
