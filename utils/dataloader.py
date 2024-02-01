import numpy as np
import pandas as pd
from scipy.stats import dirichlet
import scipy.stats as stats
from conf.settings.default import DataConfig
from utils.models import LogDataset


def synthesize_data(params: DataConfig) -> np.ndarray:
    """
    真のユーザーxアイテムの評価値行列を生成
    参考文献: How Algorithmic Confounding in Recommendation Systems
    Increases Homogeneity and Decreases Utility
    (https://arxiv.org/abs/1710.11214)

    Args:
        params (DataConfig): データの特性を決めるパラメータのインスタンス

    Returns:
        np.ndarray: 真の分布に従うユーザーxアイテムの評価値行列
    """

    mu_alpha_hat = dirichlet(
        [params.mu_u for _ in range(params.n_factors)], seed=params.seed
    ).rvs(params.n_items)
    mu_p_hat = dirichlet(
        [params.mu_i for _ in range(params.n_factors)], seed=params.seed
    ).rvs(params.n_users)

    mu_p = params.dirichlet_noise[1] * mu_p_hat
    mu_alpha = params.dirichlet_noise[0] * mu_alpha_hat

    alpha_i = np.array(
        [
            dirichlet(mu_alpha[i], seed=params.seed)
            .rvs(size=1)
            .flatten()
            .tolist()
            for i in range(params.n_items)
        ]
    )
    p_u = np.array(
        [
            dirichlet(mu_p[u], seed=params.seed).rvs(size=1).flatten().tolist()
            for u in range(params.n_users)
        ]
    )

    mu = np.dot(alpha_i, p_u.T)
    var = 0.00001

    # alphaとbetaが0以下になるのを防ぐ
    eps = 1e-5
    alpha = (((1 - mu) / (var**2)) - (1 / mu)) * (mu**2)
    alpha = np.where(alpha > 0, alpha, eps)
    beta = alpha * ((1 / mu) - 1)
    beta = np.where(beta > 0, beta, eps)

    # 真の分布を生成
    Vui = stats.beta.rvs(alpha, beta)

    return Vui.T


def generate_logged_data(params: DataConfig, Vui: np.ndarray) -> LogDataset:
    """
    各ユーザーへのランキング推薦リストを生成。さらにクリックを発生させることで人工的なログデータを作る。

    クリック発生メカニズム:
    ・trainデータ: クリック率 = 嗜好度合い率(Vui) x アイテム人気度
    ・testデータ: クリック率 = 嗜好度合い率(Vui)

    推薦リストの生成について:
    ・trainデータ: ランダム推薦
    ・testデータ: ランダム推薦
    (trainデータとtestデータが乖離している原因はクリックの発生メカニズムの違いだけであることを仮定するため)

    Args:
        params (DataConfig): データの特性を決めるパラメータのインスタンス
        Vui (np.ndarray): 真の分布に従うユーザーxアイテムの評価値行列

    Returns:
        LogDataset: ログデータのインスタンスを生成
            ・train: クリックの発生にバイアスのかかった学習データ
            ・val: クリックの発生にバイアスのかかったvalデータ(チューニング用)
            ・test: 好みが正しくラベル化されたテストデータ
            ・logged_data_matrix: ログデータ内の評価値行列
            ・pscores: 学習時に重み付けする傾向スコア
    """
    train, val, test = [], [], []
    train_size = int(params.train_test_split * params.n_rankings_per_user) - 1
    val_size = 1
    test_size = params.n_rankings_per_user - train_size

    # train_item_indices = np.arange(150)

    # 人気アイテムに露出確率を与える
    clip = 1e-7
    item_exposures = np.maximum(
        (Vui.mean(0) / Vui.mean(0).max()) ** params.p_power,
        clip,
    )

    for user_id in range(params.n_users):
        np.random.seed(params.seed + user_id)

        # trainデータを生成
        item_range = np.arange(params.n_items)
        train_items_sets = np.random.choice(
            item_range, size=(train_size, params.k), replace=False
        )

        user_ids = [user_id] * params.k
        Vu_i = Vui[user_id]
        for item_ids in train_items_sets:
            ratings = Vu_i[item_ids]
            exposures = item_exposures[item_ids]

            # P(Y = 1) = P(R = 1) * P(O = 1)
            rating_labels = np.random.binomial(n=1, p=ratings)
            exposure_labels = np.random.binomial(n=1, p=exposures)
            clicks = rating_labels * exposure_labels

            # negative sampling
            # positive_indices = np.random.choice(
            #    train_item_indices[clicks == 1],
            #    replace=False,
            #    size=params.k // 4,
            # )
            # negative_indices = np.random.choice(
            #    train_item_indices[clicks == 0],
            #    replace=False,
            #    size=params.k - len(positive_indices),
            # )
            # samples_indices = np.r_[positive_indices, negative_indices]

            click_info = np.column_stack(
                [
                    user_ids,
                    item_ids,
                    ratings,
                    exposures,
                    exposure_labels,
                    clicks
                ]
            ).tolist()
            train.append(click_info)

        # valデータを生成
        item_range = tuple(set(item_range) - set(train_items_sets.flatten()))
        val_items_sets = np.random.choice(
            item_range, size=(val_size, params.k), replace=False
        )
        for item_ids in val_items_sets:
            ratings = Vu_i[item_ids]
            exposures = item_exposures[item_ids]

            # P(Y = 1) = P(R = 1) * P(O = 1)
            rating_labels = np.random.binomial(n=1, p=ratings)
            exposure_labels = np.random.binomial(n=1, p=exposures)
            clicks = rating_labels * exposure_labels

            click_info = np.column_stack(
                [
                    user_ids,
                    item_ids,
                    ratings,
                    exposures,
                    exposure_labels,
                    clicks
                ]
            ).tolist()
            val.append(click_info)

        # testデータを生成
        item_range = tuple(
            set(item_range)
            - set(train_items_sets.flatten())
            - set(val_items_sets.flatten())
        )
        test_items_sets = np.random.choice(
            item_range, size=(test_size, params.k), replace=False
        )
        for item_ids in test_items_sets:
            ratings = Vu_i[item_ids]
            # P(Y = 1) = P(R = 1)
            ratings = np.random.binomial(n=1, p=ratings)
            click_info = np.column_stack(
                [user_ids, item_ids, ratings]
            ).tolist()
            test.append(click_info)

    train: np.ndarray = np.array(train)
    val: np.ndarray = np.array(val)
    test: np.ndarray = np.array(test)

    logged_data_matrix = _get_loggged_data_matrix(
        user_ids=train[:, :, 0].astype(int).flatten(),
        item_ids=train[:, :, 1].astype(int).flatten(),
        clicks=train[:, :, 5].astype(int).flatten(),
    )

    return LogDataset(
        train=train,
        val=val,
        test=test,
        logged_data_matrix=logged_data_matrix,
        pscores=item_exposures,
    )


def _get_loggged_data_matrix(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    clicks: np.ndarray,
) -> np.ndarray:
    """
    ログデータのユーザーxアイテムのクリック発生有無行列を生成
    出現していないユーザーxアイテムの組み合わせは、np.nanで埋める

    Args:
        user_ids (np.ndarray): ログデータ内のユーザーID配列
        item_ids (np.ndarray): ログデータ内のアイテムID配列
        clicks (np.ndarray): ログデータ内のクリックの有無配列

    Returns:
        np.ndarray: ログデータのユーザーxアイテムのクリック発生有無行列
    """
    logged_data_df = pd.DataFrame(columns=["user_id", "item_id", "is_clicked"])
    logged_data_df["user_id"] = user_ids
    logged_data_df["item_id"] = item_ids
    logged_data_df["is_clicked"] = clicks

    logged_data_df = pd.pivot_table(
        logged_data_df,
        index="user_id",
        columns="item_id",
        values="is_clicked",
    )

    columns = set(logged_data_df.columns)
    for item_id in range(item_ids.max() + 1):
        if item_id not in columns:
            logged_data_df[item_id] = np.nan

    # 列を元の順序に並び替える
    logged_data_df = logged_data_df.reindex(
        sorted(logged_data_df.columns), axis=1
    )

    return logged_data_df.values
