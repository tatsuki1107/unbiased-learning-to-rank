import numpy as np
import pandas as pd
from scipy.stats import dirichlet
import scipy.stats as stats
from conf.settings.default import DataConfig
from utils.models import TrainData, TestData, Dataset


def synthesize_data(params: DataConfig) -> np.ndarray:
    mu_p_hat = dirichlet(
        [1 for _ in range(params.n_factors)], seed=params.seed
    ).rvs(params.n_users)
    mu_alpha_hat = dirichlet(
        [100 for _ in range(params.n_factors)], seed=params.seed
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


def generate_clicks(params: DataConfig, Vui: np.ndarray) -> Dataset:
    np.random.seed(params.seed)
    train_rankings, test_rankings = [], []
    true_pscores = get_pscores(params.k, params.position_bias)

    for user_id in range(params.n_users):
        rated_items_sets = np.random.choice(
            range(0, params.n_items), size=(3, params.k), replace=False
        )
        user_ids = [user_id for _ in range(params.k)]
        for set_id, top_selected_items in enumerate(rated_items_sets):
            user_ratings = Vui[user_id, top_selected_items]

            if set_id < 2:
                clicks = np.random.binomial(
                    1, user_ratings * true_pscores, size=params.k
                )
                click_info = np.column_stack(
                    [user_ids, top_selected_items, clicks]
                )
                train_rankings.extend(list(click_info))

            else:
                true_ratings_info = np.column_stack(
                    [user_ids, top_selected_items, user_ratings]
                )
                test_rankings.extend(list(true_ratings_info))

    train_tiles, test_tiles = (
        len(train_rankings) // params.k,
        len(test_rankings) // params.k,
    )
    train_rankings, test_rankings = np.array(train_rankings), np.array(
        test_rankings
    )
    train_user_ids = train_rankings[:, 0].reshape(train_tiles, params.k)
    train_item_ids = train_rankings[:, 1].reshape(train_tiles, params.k)
    clicks = train_rankings[:, 2].reshape(train_tiles, params.k)
    test_user_ids = (
        test_rankings[:, 0].reshape(test_tiles, params.k).astype(np.int64)
    )
    test_item_ids = (
        test_rankings[:, 1].reshape(test_tiles, params.k).astype(np.int64)
    )

    ratings = test_rankings[:, 2].reshape(test_tiles, params.k)

    traindata = TrainData(train_user_ids, train_item_ids, clicks)
    testdata = TestData(test_user_ids, test_item_ids, ratings)

    logged_data_matrix = _get_loggged_data_matrix(
        train_user_ids.flatten(),
        test_user_ids.flatten(),
        train_item_ids.flatten(),
        test_item_ids.flatten(),
    )

    return Dataset(traindata, testdata, logged_data_matrix)


def _res_alpha_and_beta(mu, var=0.00001):
    alpha = (((1 - mu) / (var**2)) - (1 / mu)) * (mu**2)
    # アルファとベータが0以下になるのを防ぐ
    alpha = np.where(alpha > 0, alpha, 1e-5)
    beta = alpha * ((1 / mu) - 1)
    beta = np.where(beta > 0, beta, 1e-5)

    return alpha, beta


def get_pscores(n_positions: int, position_bias: tuple) -> np.ndarray:
    positions = np.arange(n_positions) + 1
    pscores = (position_bias[0] / positions) ** position_bias[1]
    return pscores


def _get_loggged_data_matrix(
    train_user_ids: np.ndarray,
    test_user_ids: np.ndarray,
    train_item_ids: np.ndarray,
    test_item_ids: np.ndarray,
) -> np.ndarray:
    user_ids = np.concatenate([train_user_ids, test_user_ids])
    item_ids = np.concatenate([train_item_ids, test_item_ids])

    logged_data_df = pd.DataFrame(
        columns=["user_id", "item_id", "is_logged_data"]
    )
    logged_data_df["user_id"] = user_ids
    logged_data_df["item_id"] = item_ids
    logged_data_df["is_logged_data"] = 1

    logged_data_df = pd.pivot_table(
        logged_data_df,
        index="user_id",
        columns="item_id",
        values="is_logged_data",
    )

    return logged_data_df.values
