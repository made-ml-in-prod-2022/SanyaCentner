import os
import pickle
import json
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


def validate(args):
    x_test = pd.read_csv(args.input_dir + "/x_test.csv")
    y_test = pd.read_csv(args.input_dir + "/y_test.csv")

    model = pickle.load(open(args.model_path + "/model.pkl", 'rb'))

    prediction = model.predict(x_test)

    accuracy = accuracy_score(y_test, prediction).round(3)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, prediction)

    metrics = {
        "Accurracy": accuracy,
        "Precision": precision.tolist(),
        "Recall": recall.tolist(),
        "F-score": fscore.tolist()
    }

    with open(args.model_path + "/metrics.json", "w+") as f:
        f.write(json.dumps(metrics))


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
    validate(namespace)
