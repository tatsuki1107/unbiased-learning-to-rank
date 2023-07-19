from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TrainData:
    user_ids: np.ndarray[np.ndarray[np.int64]]
    item_ids: np.ndarray[np.ndarray[np.int64]]
    clicks: np.ndarray[np.ndarray[np.int64]]


@dataclass(frozen=True)
class TestData:
    user_ids: np.ndarray[np.ndarray[np.int64]]
    item_ids: np.ndarray[np.ndarray[np.int64]]
    ratings: np.ndarray[np.ndarray[np.float64]]
