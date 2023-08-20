from dataclasses import dataclass
from typing import Tuple


@dataclass
class DataConfig:
    n_users: int = 100
    n_items: int = 500
    n_factors: int = 10
    mu_u: int = 1
    mu_i: int = 100
    dirichlet_noise: Tuple[float, float] = (0.3, 0.01)
    seed: int = 12345
    train_test_split: float = 0.8
    n_rankings_per_user: int = 10
    k: int = 10
    position_bias: Tuple[float, float] = (1.0, 0.5)
    p_power: float = 1.0
    oracle: bool = False
    is_created_dataset: bool = False
    is_created_clicks: bool = False


@dataclass
class PointwiseConfig:
    n_factors: int = 300
    scale: float = 0.01
    n_epochs: int = 10
    lr: float = 0.001
    reg: float = 1.5
    batch_size: int = 30


@dataclass
class ListwiseConfig:
    n_factors: int = 300
    scale: float = 0.01
    n_epochs: int = 10
    lr: float = 0.001
    reg: float = 1.5
    batch_size: int = 30


@dataclass
class ExperimentConfig:
    dataset: DataConfig = DataConfig()
    pointwise_config: PointwiseConfig = PointwiseConfig()
    listwise_config: ListwiseConfig = ListwiseConfig()
