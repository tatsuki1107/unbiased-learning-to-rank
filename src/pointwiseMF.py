import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from sklearn.utils import resample


@dataclass
class PointwiseRecommender:
    """Implicit Recommenders based on pointwise approach."""

    n_users: int
    n_items: int
    n_factors: int
    reg: float
    lr: float
    scale: float
    n_epochs: int
    seed: int
    n_positions: int
    oracle: bool = False
    pscores: Optional[np.ndarray] = None
    batch_size: int = 15
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self) -> None:
        """Initialize Class."""

        # initialize user-item matrices and biases
        np.random.seed(self.seed)
        self.P = np.random.normal(
            size=(self.n_users, self.n_factors),
            scale=self.scale,
        )
        self.Q = np.random.normal(
            size=(self.n_items, self.n_factors),
            scale=self.scale,
        )
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)

        self.M_P = np.zeros_like(self.P)
        self.M_Q = np.zeros_like(self.Q)
        self.V_P = np.zeros_like(self.P)
        self.V_Q = np.zeros_like(self.Q)
        self.M_item_bias = np.zeros_like(self.item_bias)
        self.V_item_bias = np.zeros_like(self.item_bias)
        self.M_user_bias = np.zeros_like(self.user_bias)
        self.V_user_bias = np.zeros_like(self.user_bias)

        if self.pscores is None:
            self.pscores = np.ones(self.n_items)

    def fit(self, dataset: tuple) -> Tuple[list]:
        train = dataset[0]
        val = dataset[1].reshape(-1, 3)

        # テストデータは評価値を使ってDCG@kを計算。好み度合い[0,1]を使って損失を計算
        test = dataset[2].reshape(-1, 4)
        self.global_bias = train[:, :, 2].mean()

        val_loss, test_loss, ndcgs = [], [], []
        for _ in range(self.n_epochs):
            samples = resample(
                train[:, :, 0],
                train[:, :, 1],
                train[:, :, 2],
                replace=True,
                n_samples=self.batch_size,
                random_state=self.seed,
            )
            for user_ids, item_ids, clicks in zip(*samples):
                user_id = user_ids[0]
                if self.oracle:
                    # 真の評価値を[0, 1]範囲にスケーリング
                    clicks = clicks / 5

                err = (clicks / self.pscores[item_ids]) - self.predict(
                    user_ids, item_ids
                )
                err = err.reshape(-1, 1)
                grad_P = (
                    np.sum(-err * self.Q[item_ids], axis=0)
                    + self.reg * self.P[user_id]
                )
                self._update_P(user_id, grad_P)

                grad_Q = -err * self.P[user_id] + self.reg * self.Q[item_ids]
                self._update_Q(item_ids, grad_Q)

                grad_user_bias = (
                    np.sum(-err) + self.reg * self.user_bias[user_id]
                )
                self._update_user_bias(user_id, grad_user_bias)

                grad_item_bias = (
                    -err.reshape(-1) + self.reg * self.item_bias[item_ids]
                )
                self._update_item_bias(item_ids, grad_item_bias)

            valloss = self._cross_entoropy_loss(
                user_ids=val[:, 0],
                item_ids=val[:, 1],
                ratings=val[:, 2],
                data="train",
            )
            val_loss.append(valloss)

            testloss = self._cross_entoropy_loss(
                user_ids=test[:, 0],
                item_ids=test[:, 1],
                ratings=test[:, 3],
                data="test",
            )
            test_loss.append(testloss)

            mean_ndcgs = self._ndcg(test)
            ndcgs.append(mean_ndcgs)

        return val_loss, test_loss, ndcgs

    def _cross_entoropy_loss(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        data: str,
    ):
        if self.oracle:
            # 真の評価値を確率に変換
            ratings = ratings / 5

        if data == "train":
            _pscores = self.pscores[item_ids]
        elif data == "test":
            _pscores = np.ones_like(item_ids)

        rogit = self.predict(user_ids, item_ids)
        loss = -np.sum(
            (ratings / _pscores) * np.log(rogit + self.eps)
            + (1 - (ratings / _pscores)) * np.log(1 - rogit + self.eps)
        ) / len(ratings)
        return loss

    def _predict_pair(self, user_id: int, item_id: int) -> float:
        return (
            np.dot(self.P[user_id], self.Q[item_id])
            + self.global_bias
            + self.item_bias[item_id]
            + self.user_bias[user_id]
        )

    def predict(
        self, user_ids: np.ndarray, item_ids: np.ndarray
    ) -> np.ndarray:
        inner_products = np.array(
            [
                self._predict_pair(user_id, item_id)
                for user_id, item_id in zip(user_ids, item_ids)
            ]
        )
        rogit = self._sigmoid(inner_products)
        return rogit

    def recommend(self, logged_data_matrix: np.ndarray):
        user_bias = np.tile(self.user_bias, (self.n_items, 1)).T
        item_bias = np.tile(self.item_bias, (self.n_users, 1))

        r_hat = self.P @ self.Q.T + self.global_bias + item_bias + user_bias
        pred_matrix = self._sigmoid(r_hat)

        rec_candidates = np.where(
            np.isnan(logged_data_matrix), pred_matrix, np.nan
        )

        recommend_list = np.argsort(-rec_candidates, axis=1, kind="mergesort")[
            :, : self.n_positions
        ]

        return recommend_list

    def _update_P(self, user_id: int, grad_P: np.ndarray):
        self.M_P[user_id] = (
            self.beta1 * self.M_P[user_id] + (1 - self.beta1) * grad_P
        )
        self.V_P[user_id] = self.beta2 * self.V_P[user_id] + (
            1 - self.beta2
        ) * (grad_P**2)
        M_P_hat = self.M_P[user_id] / (1 - self.beta1)
        V_P_hat = self.V_P[user_id] / (1 - self.beta2)
        self.P[user_id] -= self.lr * M_P_hat / ((V_P_hat**0.5) + self.eps)

    def _update_Q(self, item_ids: np.ndarray, grad_Q: np.ndarray):
        self.M_Q[item_ids] = (
            self.beta1 * self.M_Q[item_ids] + (1 - self.beta1) * grad_Q
        )
        self.V_Q[item_ids] = self.beta2 * self.V_Q[item_ids] + (
            1 - self.beta2
        ) * (grad_Q**2)
        M_Q_hat = self.M_Q[item_ids] / (1 - self.beta1)
        V_Q_hat = self.V_Q[item_ids] / (1 - self.beta2)
        self.Q[item_ids] -= self.lr * M_Q_hat / ((V_Q_hat**0.5) + self.eps)

    def _update_user_bias(self, user_id: int, grad_user_bias: np.ndarray):
        """Update user bias."""
        self.M_user_bias[user_id] = (
            self.beta1 * self.M_user_bias[user_id]
            + (1 - self.beta1) * grad_user_bias
        )
        self.V_user_bias[user_id] = self.beta2 * self.V_user_bias[user_id] + (
            1 - self.beta2
        ) * (grad_user_bias**2)
        M_user_bias_hat = self.M_user_bias[user_id] / (1 - self.beta1)
        V_user_bias_hat = self.V_user_bias[user_id] / (1 - self.beta2)
        self.user_bias[user_id] -= (
            self.lr * M_user_bias_hat / ((V_user_bias_hat**0.5) + self.eps)
        )

    def _update_item_bias(self, item_id: int, grad_item_bias: float):
        """Update item bias."""
        self.M_item_bias[item_id] = (
            self.beta1 * self.M_item_bias[item_id]
            + (1 - self.beta1) * grad_item_bias
        )
        self.V_item_bias[item_id] = self.beta2 * self.V_item_bias[item_id] + (
            1 - self.beta2
        ) * (grad_item_bias**2)
        M_item_bias_hat = self.M_item_bias[item_id] / (1 - self.beta1)
        V_item_bias_hat = self.V_item_bias[item_id] / (1 - self.beta2)
        self.item_bias[item_id] -= (
            self.lr * M_item_bias_hat / ((V_item_bias_hat**0.5) + self.eps)
        )

    def _sigmoid(self, x: np.array) -> np.array:
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-(x)))

    def _dcgs(self, user_relevances: np.ndarray, _k: int = 10) -> float:
        user_relevances = user_relevances[:, :_k]
        dcgs = user_relevances[:, 0] + np.sum(
            user_relevances[:, 1:]
            / np.log2(np.arange(2, user_relevances.shape[1] + 1)),
            axis=1,
        )
        return np.where(dcgs != 0.0, dcgs, np.nan)

    def _ndcg(self, test: np.ndarray) -> float:
        # for文使わずに行列のまま一気に計算
        scores = self.predict(user_ids=test[:, 0], item_ids=test[:, 1])
        scores = scores.reshape(-1, self.n_positions)
        ratings = test[:, 3].reshape(-1, self.n_positions)

        test_indices = np.arange(ratings.shape[0])[:, None]
        pred_ratings = ratings[
            test_indices, np.argsort(-scores, axis=1, kind="mergesort")
        ]

        return np.nanmean(self._dcgs(pred_ratings) / pred_ratings.sum(1))
