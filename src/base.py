from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple
from utils.optimizer import Adam


@dataclass
class BaseRecommender(ABC):
    """基底クラス

    args:
        n_users: ユーザ数
        n_items: アイテム数
        n_factors: MFの因子数
        reg: L2正則化パラメータ
        lr: 学習率
        scale: MFの因子の分散の初期値
        n_epochs: 学習イテレーション回数
        seed: 乱数の種
        n_positions: 各ユーザへのランキングの長さ
        pscores: 逆確率重み付けするための傾向スコア
        batch_size: 学習時のバッチサイズ
        beta1: Adamのパラメータ
        beta2: Adamのパラメータ
        eps: オーバーフロー、アンダーフローを防止するための値
    """

    n_users: int
    n_items: int
    n_factors: int
    reg: float
    lr: float
    n_epochs: int
    scale: float
    seed: int
    n_positions: int
    pscores: Optional[np.ndarray] = None
    batch_size: int = 32
    eps = 1e-8

    def __post_init__(self) -> None:
        """Initialize Class."""
        np.random.seed(self.seed)
        self.P = np.random.normal(
            size=(self.n_users, self.n_factors), scale=self.scale
        )
        self.Q = np.random.normal(
            size=(self.n_items, self.n_factors), scale=self.scale
        )
        self.adam_P = Adam(self.P.shape)
        self.adam_Q = Adam(self.Q.shape)

        if self.pscores is None:
            self.pscores = np.ones(self.n_items)

    @abstractmethod
    def fit(self, dataset: tuple) -> Tuple[list]:
        pass

    @abstractmethod
    def recommend(self, logged_data_matrix: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _cross_entoropy_loss(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        data: str,
    ) -> float:
        pass

    @abstractmethod
    def predict(self, user_ids: int, item_ids: int) -> np.ndarray:
        pass

    def _update_P(self, user_id: int, grad_P: np.ndarray) -> None:
        adam_params = self.adam_P.update(index=user_id, grad=grad_P)
        self.P[user_id] -= self.lr * adam_params

    def _update_Q(self, item_ids: np.ndarray, grad_Q: np.ndarray) -> None:
        adam_params = self.adam_Q.update(index=item_ids, grad=grad_Q)
        self.Q[item_ids] -= self.lr * adam_params
