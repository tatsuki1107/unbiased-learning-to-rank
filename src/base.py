from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple


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
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self) -> None:
        """Initialize Class."""
        np.random.seed(self.seed)
        self.P = np.random.normal(
            size=(self.n_users, self.n_factors), scale=self.scale
        )
        self.Q = np.random.normal(
            size=(self.n_items, self.n_factors), scale=self.scale
        )
        self.M_P = np.zeros_like(self.P)
        self.M_Q = np.zeros_like(self.Q)
        self.V_P = np.zeros_like(self.P)
        self.V_Q = np.zeros_like(self.Q)

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
        self.M_P[user_id] = (
            self.beta1 * self.M_P[user_id] + (1 - self.beta1) * grad_P
        )
        self.V_P[user_id] = self.beta2 * self.V_P[user_id] + (
            1 - self.beta2
        ) * (grad_P**2)
        M_P_hat = self.M_P[user_id] / (1 - self.beta1)
        V_P_hat = self.V_P[user_id] / (1 - self.beta2)
        self.P[user_id] -= self.lr * M_P_hat / ((V_P_hat**0.5) + self.eps)

    def _update_Q(self, item_ids: np.ndarray, grad_Q: np.ndarray) -> None:
        self.M_Q[item_ids] = (
            self.beta1 * self.M_Q[item_ids] + (1 - self.beta1) * grad_Q
        )
        self.V_Q[item_ids] = self.beta2 * self.V_Q[item_ids] + (
            1 - self.beta2
        ) * (grad_Q**2)
        M_Q_hat = self.M_Q[item_ids] / (1 - self.beta1)
        V_Q_hat = self.V_Q[item_ids] / (1 - self.beta2)
        self.Q[item_ids] -= self.lr * M_Q_hat / ((V_Q_hat**0.5) + self.eps)
