from src.pointwise import PointwiseRecommender
from src.listwise import ListwiseRecommender
from conf.settings.default import PointwiseConfig, DataConfig, ListwiseConfig
from utils.dataloader import generate_logged_data, synthesize_data
import numpy as np


def test_pointwise_mf():
    params = DataConfig()
    Vui = synthesize_data(params)
    dataset = generate_logged_data(params, Vui)
    model_params = PointwiseConfig(n_epochs=3)

    model = PointwiseRecommender(
        n_users=params.n_users,
        n_items=params.n_items,
        n_factors=model_params.n_factors,
        lr=model_params.lr,
        n_epochs=model_params.n_epochs,
        scale=model_params.scale,
        reg=model_params.reg,
        pscores=dataset.pscores,
        seed=12345,
        n_positions=params.k,
    )

    # click データで学習
    train = dataset.train[:, :, [0, 1, 3]]
    val = dataset.val[:, :, [0, 1, 3]]
    test = dataset.test[:, :, [0, 1, 2]]
    val_loss, test_loss, ndcgs = model.fit((train, val, test))
    recommend_list = model.recommend(dataset.logged_data_matrix)

    assert isinstance(val_loss, list)
    assert isinstance(test_loss, list)
    assert isinstance(ndcgs, list)
    assert isinstance(recommend_list, np.ndarray)


def test_listwise_mf():
    params = DataConfig()
    Vui = synthesize_data(params)
    dataset = generate_logged_data(params, Vui)
    model_params = ListwiseConfig(n_epochs=3)

    model = ListwiseRecommender(
        n_users=params.n_users,
        n_items=params.n_items,
        n_factors=model_params.n_factors,
        lr=model_params.lr,
        n_epochs=model_params.n_epochs,
        scale=model_params.scale,
        reg=model_params.reg,
        pscores=dataset.pscores,
        seed=12345,
        n_positions=params.k,
    )

    # click データで学習
    train = dataset.train[:, :, [0, 1, 3]]
    val = dataset.val[:, :, [0, 1, 3]]
    test = dataset.test[:, :, [0, 1, 2]]
    val_loss, test_loss, ndcgs = model.fit((train, val, test))
    recommend_list = model.recommend(dataset.logged_data_matrix)

    assert isinstance(val_loss, list)
    assert isinstance(test_loss, list)
    assert isinstance(ndcgs, list)
    assert isinstance(recommend_list, np.ndarray)
