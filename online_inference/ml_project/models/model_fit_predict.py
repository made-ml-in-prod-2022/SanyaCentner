import pandas as pd
import numpy as np
import pickle
import logging
from typing import Union, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

from enities import TrainingPipelineParams
from data import split_train_val_data
from features import (
    extract_target,
    CustomTransformer,
    categorical_features,
    numerical_features,
)

Model = Union[LogisticRegression, GaussianNB, CustomTransformer, OneHotEncoder]
logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def load_model(params: TrainingPipelineParams) -> Union[LogisticRegression, GaussianNB]:
    logger.info(f"Start loading {params.model_type} model")
    if params.model_type == "LogisticRegression":
        model = LogisticRegression()
    elif params.model_type == "GaussianNB":
        model = GaussianNB()
    else:
        logger.exception("There is no such model")
        raise NotImplementedError()

    logger.info("Finishing loading model")
    return model


def evaluate_model(predict: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    logger.info('Start calculate metrics for model')
    return {
        "Accuracy": metrics.accuracy_score(target, predict),
        "Roc-Auc": metrics.roc_auc_score(target, predict),
        "F1": metrics.f1_score(target, predict),
    }


def save_model(model: Model, path: str):
    logger.info(f"Saving model to {path}")
    with open(path, "wb") as f:
        pickle.dump(model, f)

    logger.info('Model saved')


def save_transformer(model: Model, path: str):
    logger.info(f"Saving transformer to {path}")
    with open(path, "wb") as f:
        pickle.dump(model, f)

    logger.info('Transformer saved')


def save_encoder(model: Model, path: str):
    logger.info(f"Saving encoder to {path}")
    with open(path, "wb") as f:
        pickle.dump(model, f)

    logger.info('Encoder saved')


def open_model(path: str) -> Model:
    logger.info(f"Opening model {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def open_transformer(path: str) -> Model:
    logger.info(f"Opening transformer {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def open_encoder(path: str) -> Model:
    logger.info(f"Opening encoder {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def run_train_pipeline(params: TrainingPipelineParams) -> Dict[str, float]:
    data = pd.read_csv(params.input_train_data_path)
    data, target = extract_target(data, params)

    transformer = CustomTransformer(params.features.numerical_features)
    transformer.fit(data)
    save_transformer(transformer, params.transformer_path)

    logger.info(f"start train pipeline with params {params}")

    train_data, val_data, train_target, val_target = split_train_val_data(
        data, target, params
    )

    onehotenc = OneHotEncoder()
    onehotenc.fit(data[params.features.categorical_features])
    save_encoder(onehotenc, params.onehotenc_path)

    df_categ = categorical_features(train_data, onehotenc, params)
    df_num = numerical_features(train_data, transformer, params)
    train_data_result = pd.concat([df_categ, df_num], axis=1)

    model = load_model(params)
    logger.info('Start model fitting')
    model.fit(train_data_result, train_target)
    save_model(model, params.model_path)

    val_df_categ = categorical_features(val_data, onehotenc, params)
    val_df_num = numerical_features(val_data, transformer, params)
    val_data_result = pd.concat([val_df_categ, val_df_num], axis=1)

    predict = model.predict(val_data_result)
    result_metrics = evaluate_model(predict, val_target)
    for key in result_metrics:
        print(key, " : ", result_metrics[key])

    return result_metrics


def run_test_pipeline(params: TrainingPipelineParams):
    target_col = params.features.target_col
    data = pd.read_csv(params.input_test_data_path)

    transformer = open_transformer(params.transformer_path)
    onehotenc = open_encoder(params.onehotenc_path)
    model = open_model(params.model_path)

    test_df_categ = categorical_features(data, onehotenc, params)
    test_df_num = numerical_features(data, transformer, params)
    test_data = pd.concat([test_df_categ, test_df_num], axis=1)

    predict = model.predict(test_data)

    answer = pd.read_csv(params.input_train_data_path)[target_col]

    result_metrics = evaluate_model(predict, answer)

    for key in result_metrics:
        print(key, " : ", result_metrics[key])

    logger.info(f"Saving predict to {params.predict_path}")
    predict_df = pd.DataFrame({target_col: predict})
    predict_df.to_csv(params.predict_path, index=False)

    return result_metrics
