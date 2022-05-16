import pandas as pd
import numpy as np
import logging
from typing import List, Tuple

from enities import TrainingPipelineParams

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

def extract_target(
    data: pd.DataFrame,
    params: TrainingPipelineParams
) -> Tuple[pd.DataFrame, np.ndarray]:
    target_col = params.features.target_col
    target = data[target_col].values
    data = data.drop(columns=[target_col])
    return data, target
