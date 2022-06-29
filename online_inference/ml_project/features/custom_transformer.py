
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)

from enities import TrainingPipelineParams


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features):
        self.scaler = StandardScaler()
        self.numerical_features = numerical_features

    def fit(self, data: pd.DataFrame):
        self.scaler.fit(data[self.numerical_features])
        return self

    def transform(self, data: pd.DataFrame):
        data_new = data[self.numerical_features].copy()
        data_new = self.scaler.transform(data_new)
        return data_new


def categorical_features(
    data: pd.DataFrame,
    ohe: OneHotEncoder,
    params: TrainingPipelineParams
) -> pd.DataFrame:
    cat_columns = params.features.categorical_features
    cat_df = pd.DataFrame(ohe.transform(data[cat_columns]).toarray())
    cat_df.rename(columns=str, inplace=True)
    return cat_df


def numerical_features(
    data: pd.DataFrame,
    transformer: CustomTransformer,
    params: TrainingPipelineParams
) -> pd.DataFrame:
    num_features = params.features.numerical_features
    num_df = pd.DataFrame(
        columns=num_features, data=transformer.transform(data[num_features])
    )
    return num_df
