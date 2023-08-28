from dataclasses import dataclass
import numpy as np
from typing import Tuple, Union
from sklearn.utils import resample
from utils.metrics import calc_dcg
from src.base import BaseRecommender


@dataclass
class ListwiseRecommender(BaseRecommender):
    """matrix factorizationにTop-1 リストワイズ損失を用いた推薦モデルのクラス"""

    def __post_init__(self) -> None:
        """Initialize Class."""

        # initialize user-item matrices and biases
        super(ListwiseRecommender, self).__post_init__()

    def fit(self, dataset: tuple) -> Tuple[list]:
        train = dataset[0]
        val = dataset[1].reshape(-1, 3)
        test = dataset[2].reshape(-1, 3)

        val_pscores = self.pscores[val[:, 1].astype(np.int64)]
        test_pscores = np.ones_like(test[:, 1])

        train_loss, val_loss, test_loss, test_ndcgs = [], [], [], []
        for _ in range(self.n_epochs):
            samples = resample(
                train[:, :, 0].astype(np.int64),
                train[:, :, 1].astype(np.int64),
                train[:, :, 2],
                replace=True,
                n_samples=self.batch_size,
                random_state=self.seed,
            )
            for user_ids, item_ids, clicks in zip(*samples):
                user_id = user_ids[0]
                softmax = self._softmax(self.Q[item_ids] @ self.P[user_id])
                grad_P = (
                    -(
                        (clicks / self.pscores[item_ids])
                        @ (self.Q[item_ids] - softmax @ self.Q[item_ids])
                    )
                    + self.reg * self.P[user_id]
                )
                self._update_P(user_id, grad_P)

                grad_Q = (
                    -((clicks / self.pscores[item_ids]) * (1 - softmax))[
                        :, None
                    ]
                    * self.P[user_id]
                    + self.reg * self.Q[item_ids]
                )
                self._update_Q(item_ids, grad_Q)

            trainloss = self._cross_entoropy_loss(
                user_ids=samples[0].reshape(-1),
                item_ids=samples[1].reshape(-1),
                ratings=samples[2].reshape(-1),
                pscores=self.pscores[samples[1].reshape(-1)],
            )
            train_loss.append(trainloss)

            valloss = self._cross_entoropy_loss(
                user_ids=val[:, 0].astype(np.int64),
                item_ids=val[:, 1].astype(np.int64),
                ratings=val[:, 2],
                pscores=val_pscores,
            )
            val_loss.append(valloss)

            testloss = self._cross_entoropy_loss(
                user_ids=test[:, 0],
                item_ids=test[:, 1],
                ratings=test[:, 2],
                pscores=test_pscores,
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

        return train_loss, val_loss, test_loss, test_ndcgs

    def _cross_entoropy_loss(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        pscores: np.ndarray,
    ) -> float:
        softmax = self.predict(user_ids, item_ids)
        loss = -np.sum((ratings / pscores) * np.log(softmax + self.eps)) / len(
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
