from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LogDataset:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    logged_data_matrix: np.ndarray
    pscores: np.ndarray
