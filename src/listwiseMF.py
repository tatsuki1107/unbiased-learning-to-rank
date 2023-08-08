from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple
from utils.models import TrainData, TestData
from sklearn.utils import resample


@dataclass
class ListwiseRecommender:
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
            scale=0.0001,
        )
        self.Q = np.random.normal(
            size=(self.n_items, self.n_factors),
            scale=0.0001,
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
            for user_ids, item_ids, clicks in zip(*samples):
                user_id = user_ids[0]
                softmax = self.plackett_luce(
                    self.Q[item_ids] @ self.P[user_id]
                )
                grad_P = (
                    -(
                        (clicks / self.pscores)
                        @ (self.Q[item_ids] - softmax @ self.Q[item_ids])
                    )
                    + self.reg * self.P[user_id]
                )
                self._update_P(user_id, grad_P)

                grad_Q = (
                    -((clicks / self.pscores) * (1 - softmax))[:, None]
                    * self.P[user_id]
                    + self.reg * self.Q[item_ids]
                )
                self._update_Q(item_ids, grad_Q)

        return train_loss, test_loss, ndcgs

    def _cross_entoropy_loss(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        pscores: np.ndarray,
    ):
        softmax = self.predict(user_ids, item_ids)
        loss = -np.sum((ratings / pscores) * np.log(softmax)) / len(ratings)
        return loss

    def _softmax(self, x):
        x = x - np.max(x)  # オーバーフロー防止
        return np.exp(x) / np.sum(np.exp(x))

    def plackett_luce(self, arr):
        if arr.ndim == 1:
            return np.array(
                [self._softmax(arr[i:])[0] for i in range(len(arr))]
            )
        elif arr.ndim == 2:
            return np.array(
                [
                    [self._softmax(arr[j, i:])[0] for i in range(arr.shape[1])]
                    for j in range(arr.shape[0])
                ]
            )

    def predict(self, user_ids: int, item_ids: int):
        inner_products = np.array(
            [
                np.dot(self.P[user_id], self.Q[item_id])
                for user_id, item_id in zip(user_ids, item_ids)
            ]
        ).reshape(-1, self.n_positions)
        softmax = self.plackett_luce(inner_products).flatten()
        return softmax

    def recommend(self, logged_data_matrix: np.ndarray):
        pred_matrix = self.P @ self.Q.T
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

    def _dcg(self, user_relevances: np.ndarray, _k: int = 3) -> float:
        user_relevances = user_relevances[:_k]
        if len(user_relevances) == 0:
            return 0.0
        return user_relevances[0] + np.sum(
            user_relevances[1:]
            / np.log2(np.arange(2, len(user_relevances) + 1))
        )

    def _ndcg(self, test: TestData) -> float:
        ndcgs = []
        for user_ids, item_ids, ratings in zip(
            test.user_ids, test.item_ids, test.ratings
        ):
            # indcg = self._dcg(np.sort(ratings)[::-1])
            # if not indcg:
            #    ndcgs.append(0.0)

            scores = self.predict(user_ids, item_ids)
            pred_ratings = ratings[scores.argsort()[::-1]]
            dcg = self._dcg(pred_ratings)
            ndcgs.append(dcg / np.sum(pred_ratings))

        return np.mean(ndcgs)
