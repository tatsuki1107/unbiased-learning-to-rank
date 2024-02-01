from dataclasses import dataclass
import numpy as np
from typing import Tuple, Union
from sklearn.utils import resample
from utils.metrics import calc_dcg
from src.base import BaseRecommender


@dataclass
class ListwiseRecommender(BaseRecommender):
    """
    ListRank-MF (https://dl.acm.org/doi/abs/10.1145/1864708.1864764
    ?casa_token=MUgYve_rEOoAAAAA:j1ljuuHOeQ3ic8s_dtv5xSA2SLZQbQUio74J
    CfCoob5YDSdPCxQkANgLY2RRiqOF0xzYHcQghUR5) をunbiasedに拡張した新たな推定量
    """

    def __post_init__(self) -> None:
        """Initialize Class."""

        # initialize user-item matrices and biases
        super(ListwiseRecommender, self).__post_init__()

    def fit(self, dataset: tuple) -> Tuple[list]:
        train = dataset[0]
        val = dataset[1].reshape(-1, 3)
        test = dataset[2].reshape(-1, 3)

        val_loss, test_loss, test_ndcgs = [], [], []
        for epoch in range(self.n_epochs):
            samples = resample(
                train[:, :, 0].astype(np.int64),
                train[:, :, 1].astype(np.int64),
                train[:, :, 2],
                replace=False,
                n_samples=self.batch_size,
                random_state=epoch,
            )
            for user_ids, item_ids, labels in zip(*samples):
                user_id = user_ids[0]
                softmax = self._softmax(self.Q[item_ids] @ self.P[user_id])
                grad_P = (
                    -(
                        labels @
                        (self.Q[item_ids] - softmax @ self.Q[item_ids])
                    )
                    + self.reg * self.P[user_id]
                )
                self._update_P(user_id, grad_P)

                grad_Q = (
                    -(labels * (1 - softmax))[:, None]
                    * self.P[user_id]
                    + self.reg * self.Q[item_ids]
                )
                self._update_Q(item_ids, grad_Q)

            valloss = self._cross_entoropy_loss(
                user_ids=val[:, 0].astype(np.int64),
                item_ids=val[:, 1].astype(np.int64),
                ratings=val[:, 2],
            )
            val_loss.append(valloss)

            testloss = self._cross_entoropy_loss(
                user_ids=test[:, 0],
                item_ids=test[:, 1],
                ratings=test[:, 2],
            )
            test_loss.append(testloss)

            test_scores = self.predict(
                user_ids=test[:, 0],
                item_ids=test[:, 1],
            )
            mean_ndcgs = calc_dcg(
                ratings=test[:, 2].reshape(-1, self.n_positions),
                scores=test_scores.reshape(-1, self.n_positions),
            )
            test_ndcgs.append(mean_ndcgs)

        return val_loss, test_loss, test_ndcgs

    def _cross_entoropy_loss(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
    ) -> float:
        softmax = self.predict(user_ids, item_ids)
        loss = -np.sum(ratings * np.log(softmax + self.eps)) / len(
            ratings
        )
        return loss

    def _softmax(self, x: np.ndarray) -> Union[float, np.ndarray]:
        # xが1次元の場合と2次元の場合で処理を分ける
        if len(x.shape) == 1:
            x = x - np.max(x)
            return np.exp(x) / np.sum(np.exp(x))
        elif len(x.shape) == 2:
            x = x - np.max(x, axis=1, keepdims=True)
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        else:
            raise ValueError("Input array should be 1D or 2D.")

    def predict(self, user_ids: int, item_ids: int) -> np.ndarray:
        inner_products = np.array(
            [
                np.dot(self.P[user_id], self.Q[item_id])
                for user_id, item_id in zip(user_ids, item_ids)
            ]
        ).reshape(-1, self.n_positions)
        softmax = self._softmax(inner_products).flatten()
        return softmax

    def recommend(self, logged_data_matrix: np.ndarray) -> np.ndarray:
        pred_matrix = self.P @ self.Q.T
        rec_candidates = np.where(
            np.isnan(logged_data_matrix), pred_matrix, np.nan
        )

        recommend_list = np.argsort(-rec_candidates, axis=1, kind="mergesort")[
            :, : self.n_positions
        ]

        return recommend_list
