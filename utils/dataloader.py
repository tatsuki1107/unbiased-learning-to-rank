import numpy as np
import pandas as pd
from scipy.stats import dirichlet
import scipy.stats as stats
from conf.settings.default import DataConfig
from utils.models import LogDataset
from sklearn.model_selection import train_test_split


def synthesize_data(params: DataConfig) -> np.ndarray:
    mu_p_hat = dirichlet(
        [params.mu_u for _ in range(params.n_factors)], seed=params.seed
    ).rvs(params.n_users)
    mu_alpha_hat = dirichlet(
        [params.mu_i for _ in range(params.n_factors)], seed=params.seed
    ).rvs(params.n_items)

    mu_p = params.dirichlet_noise[0] * mu_p_hat
    mu_alpha = params.dirichlet_noise[1] * mu_alpha_hat

    p_u = np.array(
        [
            dirichlet(mu_p[u], seed=params.seed).rvs(size=1).flatten().tolist()
            for u in range(params.n_users)
        ]
    )
    alpha_i = np.array(
        [
            dirichlet(mu_alpha[i], seed=params.seed)
            .rvs(size=1)
            .flatten()
            .tolist()
            for i in range(params.n_items)
        ]
    )

    mu = np.dot(p_u, alpha_i.T)
    alpha, beta = _res_alpha_and_beta(mu)

    # 真の分布を生成
    Vui = stats.beta.rvs(alpha, beta)

    return Vui


def generate_logged_data(params: DataConfig, Vui: np.ndarray) -> LogDataset:
    train, test = [], []
    # true_pscores = get_pscores(params.k, params.position_bias)
    train_size = int(params.train_test_split * params.n_rankings_per_user)
    test_size = params.n_rankings_per_user - train_size

    for user_id in range(params.n_users):
        np.random.seed(params.seed + user_id)

        # create not random and positive unlabeld train data
        item_range = np.arange(params.n_items)
        train_items_sets = get_items_depends_on_policy(
            item_range, params.k, params.policy, Vui, user_id, train_size
        )
        user_ids = [user_id] * params.k
        Vu_i = Vui[user_id]
        for items in train_items_sets:
            ratings = Vu_i[items]

            clicks = np.where(ratings >= 0.75, 1, 0).astype(np.int64)
            # clicks = np.random.binomial(n=1, p=ratings * true_pscores)
            ratings = (np.floor(ratings * 5) + 1).astype(np.int64)

            click_info = np.column_stack(
                [user_ids, items, ratings, clicks]
            ).tolist()
            train.append(click_info)

        # create random and positive unlabeld test data
        item_range = tuple(set(item_range) - set(train_items_sets.flatten()))
        test_items_sets = get_items_depends_on_policy(
            item_range, params.k, "random", Vui, user_id, test_size
        )
        for items in test_items_sets:
            ratings = Vu_i[items]

            # テストデータは真の分布に従うので、好き嫌いのラベル化が可能
            clicks = np.where(ratings >= 0.75, 1, 0).astype(np.int64)
            ratings = (np.floor(ratings * 5) + 1).astype(np.int64)
            click_info = np.column_stack(
                [user_ids, items, ratings, clicks]
            ).tolist()
            test.append(click_info)

    train: np.ndarray = np.array(train)
    test: np.ndarray = np.array(test)

    logged_data_matrix = _get_loggged_data_matrix(
        user_ids=train[:, :, 0].flatten(),
        item_ids=train[:, :, 1].flatten(),
        Vui=Vui,
    )

    _train = train.copy().reshape(-1, 4)
    appeared_item_ids, item_freqs = np.unique(
        _train[_train[:, 3] == 1, 1], return_counts=True
    )
    del _train

    item_freqs_prob = item_freqs / item_freqs.max()
    item_freqs_dict = dict(zip(appeared_item_ids, item_freqs_prob))

    # ラベル1のないアイテムにはitem_freqs_prob.min() にさらに0.1倍したものを与える
    pscores = np.ones(params.n_items) * item_freqs_prob.min() * 0.1
    for item_id, pscore in item_freqs_dict.items():
        pscores[item_id] = pscore**params.p_power

    train, val = train_test_split(
        train, test_size=0.3, random_state=params.seed
    )

    return LogDataset(
        train=train,
        val=val,
        test=test,
        logged_data_matrix=logged_data_matrix,
        pscores=pscores,
    )


def _res_alpha_and_beta(mu, var=0.00001):
    alpha = (((1 - mu) / (var**2)) - (1 / mu)) * (mu**2)
    # アルファとベータが0以下になるのを防ぐ
    alpha = np.where(alpha > 0, alpha, 1e-5)
    beta = alpha * ((1 / mu) - 1)
    beta = np.where(beta > 0, beta, 1e-5)

    return alpha, beta


def get_items_depends_on_policy(
    item_range: int,
    k: int,
    policy: str,
    Vui: np.ndarray,
    user_id: int,
    datasize: int,
):
    if policy == "random":
        items_sets = np.random.choice(
            item_range, size=(datasize, k), replace=False
        )
    elif policy == "selection_bias":
        bias = 500
        p = np.exp(Vui[user_id] * bias) / np.sum(np.exp(Vui[user_id] * bias))
        items_sets = np.random.choice(
            item_range,
            size=(datasize, k),
            p=p,
            replace=False,
        )

    else:
        raise NotImplementedError

    return items_sets


def get_pscores(n_positions: int, position_bias: tuple) -> np.ndarray:
    positions = np.arange(n_positions) + 1
    pscores = (position_bias[0] / positions) ** position_bias[1]
    return pscores


def _get_loggged_data_matrix(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    Vui: np.ndarray,
) -> np.ndarray:
    logged_data_df = pd.DataFrame(
        columns=["user_id", "item_id", "is_logged_data"]
    )
    logged_data_df["user_id"] = user_ids
    logged_data_df["item_id"] = item_ids
    logged_data_df["is_logged_data"] = Vui[user_ids, item_ids]

    logged_data_df = pd.pivot_table(
        logged_data_df,
        index="user_id",
        columns="item_id",
        values="is_logged_data",
    )

    columns = set(logged_data_df.columns)
    for item_id in range(Vui.shape[1]):
        if item_id not in columns:
            logged_data_df[item_id] = np.nan

    # 列を元の順序に並び替える
    logged_data_df = logged_data_df.reindex(
        sorted(logged_data_df.columns), axis=1
    )

    return logged_data_df.values
