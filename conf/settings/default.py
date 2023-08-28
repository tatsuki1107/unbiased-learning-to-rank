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
    n_rankings_per_user: int = 5
    k: int = 15
    p_power: float = 1.0
    is_created_dataset: bool = False
    is_created_clicks: bool = False


@dataclass
class PointwiseConfig:
    n_factors: int = 30
    scale: float = 0.0012754980286032412
    n_epochs: int = 94
    lr: float = 0.021675098379053206
    reg: float = 0.0005926736508739346
    batch_size: int = 197


@dataclass
class ListwiseConfig:
    n_factors: int = 43
    scale: float = 0.0012754980286032412
    n_epochs: int = 94
    lr: float = 0.010255764352948631
    reg: float = 0.3517244890698348
    batch_size: int = 195


@dataclass
class ExperimentConfig:
    dataset: DataConfig = DataConfig()
    pointwise_config: PointwiseConfig = PointwiseConfig()
    listwise_config: ListwiseConfig = ListwiseConfig()
