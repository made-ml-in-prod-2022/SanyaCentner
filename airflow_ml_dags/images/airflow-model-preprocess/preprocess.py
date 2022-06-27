import os
import numpy as np
import pandas as pd
import argparse

def preprocess(args):

    data = pd.read_csv(args.data_file + "/data.csv")
    target = pd.read_csv(args.data_file + "/target.csv")

    data = (data - data.mean()) / data.std()

    data['target'] = target['Target'].values

    with open(args.otput_file, "w+") as f:
        f.write(data.to_csv(index=False))

def createparser():
    """read argument"""
    parser_args = argparse.ArgumentParser()
    parser_args.add_argument(
        "--data", type=str, default="data.csv", help="input data file path", dest="data_file"
    )
    parser_args.add_argument(
        "--processed", type=str, default="train_data.csv", help="train data file path", dest="otput_file"
    )
    return parser_args


if __name__ == '__main__':
    parser = createparser()
    namespace = parser.parse_args()
    preprocess(namespace)