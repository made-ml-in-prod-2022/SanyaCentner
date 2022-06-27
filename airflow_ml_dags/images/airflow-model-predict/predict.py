import os
import pickle
import pandas as pd
import argparse


def predict(args):
    data = pd.read_csv(args.data_file)

    model = pickle.load(open(args.model_path, 'rb'))

    pred = model.predict(data)

    with open(args.otput_file, "w+") as f:
        f.write(pd.DataFrame(pred, columns=["prediction"]).to_csv(index=False))


def createparser():
    """read argument"""
    parser_args = argparse.ArgumentParser()
    parser_args.add_argument(
        "--data", type=str, default="data.csv", help="input data file path", dest="data_file"
    )
    parser_args.add_argument(
        "--model", type=str, default="model.pkl", help="model file path", dest="model_path"
    )
    parser_args.add_argument(
        "--predict", type=str, default="predictions.csv", help="predicts file path", dest="otput_file"
    )
    return parser_args


if __name__ == '__main__':
    parser = createparser()
    namespace = parser.parse_args()
    predict(namespace)
