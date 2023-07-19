import numpy as np
from dataclasses import dataclass
from utils.models import TrainData, TestData
from typing import Optional, List, Tuple
from sklearn.utils import resample


@dataclass
class PointwiseRecommender:
    """Implicit Recommenders based on pointwise approach."""

    n_users: int
    n_items: int
    n_factors: int
    reg: float
    lr: float
    n_epochs: int
    seed: int
    n_positions: int
    pscores: Optional[np.ndarray] = None
    batch_size: int = 32
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self) -> None:
        """Initialize Class."""

        # initialize user-item matrices and biases
        np.random.seed(self.seed)
        self.P = np.random.normal(
            size=(self.n_users, self.n_factors),
            scale=1 / np.sqrt(self.n_factors),
        )
        self.Q = np.random.normal(
            size=(self.n_items, self.n_factors),
            scale=1 / np.sqrt(self.n_factors),
        )
        self.M_P = np.zeros_like(self.P)
        self.M_Q = np.zeros_like(self.Q)
        self.V_P = np.zeros_like(self.P)
        self.V_Q = np.zeros_like(self.Q)

        if self.pscores is None:
            self.pscores = np.ones(self.n_positions)

    def fit(self, dataset: Tuple[TrainData, TestData]):
        train, test = dataset[0], dataset[1]
        train_tiled_pscores = np.tile(self.pscores, self.batch_size)
        test_tiled_pscores = np.ones(len(self.pscores) * len(test.ratings))

        train_loss, test_loss, ndcgs = [], [], []
        for _ in range(self.n_epochs):
            samples = resample(
                train.user_ids,
                train.item_ids,
                train.clicks,
                replace=True,
                n_samples=self.batch_size,
                random_state=self.seed,
            )
            for user_ids, item_ids, clicks in zip(*samples):
                user_id = user_ids[0]
                err = (clicks / self.pscores) - self._sigmoid(
                    self.Q[item_ids] @ self.P[user_id]
                )
                err = err.reshape(-1, 1)
                grad_P = (
                    np.sum(-err * self.Q[item_ids], axis=0)
                    + self.reg * self.P[user_id]
                )
                self._update_P(user_id, grad_P)
                grad_Q = -err * self.P[user_id] + self.reg * self.Q[item_ids]
                self._update_Q(item_ids, grad_Q)

            trainloss = self._cross_entoropy_loss(
                samples[0].reshape(-1),
                samples[1].reshape(-1),
                samples[2].reshape(-1),
                train_tiled_pscores,
            )
            train_loss.append(trainloss)

            testloss = self._cross_entoropy_loss(
                test.user_ids.reshape(-1),
                test.item_ids.reshape(-1),
                test.ratings.reshape(-1),
                test_tiled_pscores,
            )
            test_loss.append(testloss)

            mean_ndcgs = self._ndcg(test)
            ndcgs.append(mean_ndcgs)

        return train_loss, test_loss, ndcgs

    def _cross_entoropy_loss(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        pscores: np.ndarray,
    ):
        rogit = self.predict(user_ids, item_ids)
        loss = -np.sum(
            (ratings / pscores) * np.log(rogit)
            + (1 - (ratings / pscores)) * np.log(1 - rogit + self.eps)
        ) / len(ratings)
        return loss

    def predict(self, user_ids: int, item_ids: int):
        inner_products = np.array(
            [
                np.dot(self.P[user_id], self.Q[item_id])
                for user_id, item_id in zip(user_ids, item_ids)
            ]
        )
        rogit = self._sigmoid(inner_products)
        return rogit

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

    def _sigmoid(self, x: np.array) -> np.array:
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def _dcg(self, scores: List[float]) -> float:
        return np.sum(
            [
                (np.power(2, score) - 1) / np.log2(i + 2)
                for i, score in enumerate(scores)
            ]
        )

    def _ndcg(self, test: TestData):
        ndcgs = []
        for user_ids, item_ids, ratings in zip(
            test.user_ids, test.item_ids, test.ratings
        ):
            pred_scores = self.predict(user_ids, item_ids)
            ranked_r = ratings[np.argsort(-pred_scores)]
            dcg = self._dcg(ranked_r)
            idcg = self._dcg(np.sort(ratings)[::-1])

            if idcg != 0:
                ndcgs.append(dcg / idcg)

        return np.mean(ndcgs)
