from marshmallow_dataclass import class_schema
from dataclasses import dataclass
import yaml

from .feature_params import Features
from .split_params import SplittingParams


@dataclass()
class TrainingPipelineParams:
    input_train_data_path: str
    input_test_data_path: str
    transformer_path: str
    onehotenc_path: str
    model_path: str
    predict_path: str
    model_type: str
    features: Features
    splitting_params: SplittingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))

