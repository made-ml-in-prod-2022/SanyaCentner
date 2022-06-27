import os
import pandas as pd
import argparse
import pickle
from sklearn.linear_model import LogisticRegression


def train(args):
    x_train = pd.read_csv(args.input_dir + "/x_train.csv")
    y_train = pd.read_csv(args.input_dir + "/y_train.csv")

    model = LogisticRegression()

    model.fit(x_train, y_train)

    with open(os.path.join(args.model_path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


def createparser():
    """read argument"""
    parser_args = argparse.ArgumentParser()
    parser_args.add_argument(
        "--split", type=str, help="split data dir path", dest="input_dir"
    )
    parser_args.add_argument(
        "--model", type=str, help="model file path", dest="model_path"
    )
    return parser_args


if __name__ == '__main__':
    parser = createparser()
    namespace = parser.parse_args()
    train(namespace)
