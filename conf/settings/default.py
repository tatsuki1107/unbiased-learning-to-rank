from dataclasses import dataclass
from typing import Tuple


@dataclass
class DataConfig:
    n_users: int = 10000
    n_items: int = 15000
    n_factors: int = 10
    dirichlet_noise: Tuple[float, float] = (0.3, 0.01)
    seed: int = 12345
    k: int = 10
    position_bias: Tuple[float, float] = (0.9, 1.0)
    is_created_dataset: bool = False
    is_created_clicks: bool = False


@dataclass
class PointwiseConfig:
    n_factors: int = 300
    n_epochs: int = 10
    lr: float = 0.001
    reg: float = 1.5


@dataclass
class ListwiseConfig:
    n_factors: int = 300
    n_epochs: int = 10
    lr: float = 0.001
    reg: float = 1.5


@dataclass
class ExperimentConfig:
    dataset: DataConfig = DataConfig()
    pointwise_config: PointwiseConfig = PointwiseConfig()
    listwise_config: ListwiseConfig = ListwiseConfig()
