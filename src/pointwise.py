import numpy as np
from dataclasses import dataclass
from typing import Tuple
from sklearn.utils import resample
from utils.metrics import calc_ndcg
from base import BaseRecommender


@dataclass
class PointwiseRecommender(BaseRecommender):
    """matrix factorizationにポイントワイズ損失を用いた推薦モデルのクラス
    pscoresがあれば、Relevance-MF (https://arxiv.org/pdf/1909.03601.pdf)

    pscoresがなければ、 Probabilistic matrix factorization
    (https://proceedings.neurips.cc/paper_files/paper/2007/file/d7322ed717dedf1eb4e6e52a37ea7bcd-Paper.pdf)
    """

    def __post_init__(self) -> None:
        """各パラメータの初期化"""

        super(PointwiseRecommender, self).__post_init__()
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)

        self.M_item_bias = np.zeros_like(self.item_bias)
        self.V_item_bias = np.zeros_like(self.item_bias)
        self.M_user_bias = np.zeros_like(self.user_bias)
        self.V_user_bias = np.zeros_like(self.user_bias)

    def fit(self, dataset: tuple) -> Tuple[list]:
        """学習後にエポックごとの損失と評価指標を求める

        Args:
            dataset (tuple): train, val, testデータを含むタプル

        Returns:
            Tuple[list]: _description_
        """
        train = dataset[0]
        val = dataset[1].reshape(-1, 3)
        test = dataset[2].reshape(-1, 3)
        self.global_bias = train[:, :, 2].mean()

        val_loss, test_loss, test_ndcgs = [], [], []
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
                user_ids=val[:, 0].astype(np.int64),
                item_ids=val[:, 1].astype(np.int64),
                ratings=val[:, 2],
                data="train",
            )
            val_loss.append(valloss)

            testloss = self._cross_entoropy_loss(
                user_ids=test[:, 0],
                item_ids=test[:, 1],
                ratings=test[:, 2],
                data="test",
            )
            test_loss.append(testloss)

            test_scores = self.predict(
                user_ids=test[:, 0],
                item_ids=test[:, 1],
            ).reshape(-1, self.n_positions)
            mean_ndcgs = calc_ndcg(
                ratings=test[:, 2].reshape(-1, self.n_positions),
                scores=test_scores,
            )
            test_ndcgs.append(mean_ndcgs)

        return val_loss, test_loss, test_ndcgs

    def _cross_entoropy_loss(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        data: str,
    ) -> float:
        """エポック毎のエントロピーロスを算出

        Args:
            data (str): "test"データは真の分布に従うため、重み付けをする必要が
            ない。よって、_pscoresをすべて1にするための引数
        """

        if data == "train":
            _pscores = self.pscores[item_ids]
        elif data == "test":
            _pscores = np.ones_like(item_ids)

        pred_ratings = self.predict(user_ids, item_ids)
        loss = -np.sum(
            (ratings / _pscores) * np.log(pred_ratings + self.eps)
            + (1 - (ratings / _pscores)) * np.log(1 - pred_ratings + self.eps)
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
        pred_ratings = self._sigmoid(inner_products)
        return pred_ratings

    def recommend(self, logged_data_matrix: np.ndarray) -> np.ndarray:
        """学習後のパラメータを使い次のレコメンドリストを生成

        Args:
            logged_data_matrix (np.ndarray): ログデータのユーザーxアイテムの
            クリックの発生有無行列。np.nanの要素だけが次推薦リストに載る候補

        Returns:
            recommend_list (np.ndarray): ユーザー数xランキングの長さの配列。要素はアイテムのID
        """
        user_bias = np.tile(self.user_bias, (self.n_items, 1)).T
        item_bias = np.tile(self.item_bias, (self.n_users, 1))

        pred_ratings = (
            self.P @ self.Q.T + self.global_bias + item_bias + user_bias
        )
        pred_matrix = self._sigmoid(pred_ratings)

        rec_candidates = np.where(
            np.isnan(logged_data_matrix), pred_matrix, np.nan
        )

        recommend_list = np.argsort(-rec_candidates, axis=1, kind="mergesort")[
            :, : self.n_positions
        ]

        return recommend_list

    def _update_user_bias(
        self, user_id: int, grad_user_bias: np.ndarray
    ) -> None:
        """ユーザーバイアス項の更新"""

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

    def _update_item_bias(self, item_id: int, grad_item_bias: float) -> None:
        """アイテムバイアス項の更新"""

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
        return 1 / (1 + np.exp(-x))
