from sklearn.model_selection import train_test_split
from typing import Tuple
import pandas as pd
import numpy as np
import logging

from enities import TrainingPipelineParams

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def split_train_val_data(
    data: pd.DataFrame,
    target: np.ndarray,
    params: TrainingPipelineParams
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    val_size = params.splitting_params.val_size
    random_state = params.splitting_params.random_state
    stratify = params.splitting_params.stratify

    logger.info(
        f"Splitting parameters: val_size = {val_size}, "
        f"random_state = {random_state}, stratify = {stratify}"
    )

    if stratify:
        train_data, val_data, train_target, val_target = train_test_split(
            data, target, test_size=val_size, random_state=random_state, stratify=target
        )
    else:
        train_data, val_data, train_target, val_target = train_test_split(
            data, target, test_size=val_size, random_state=random_state
        )

    return train_data, val_data, train_target, val_target