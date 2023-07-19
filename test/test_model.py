from src.pointwiseMF import PointwiseRecommender
from src.listwiseMF import ListwiseRecommender
from conf.settings.default import ModelConfig
from utils.models import TrainData, TestData
from utils.dataloader import get_pscores
import numpy as np


def test_pointwise_mf():
    n_users = 2
    n_items = 4
    k = 2
    position_bias = (0.9, 1.0)
    estimated_pscores = get_pscores(k, position_bias)
    model_params = ModelConfig()
    model = PointwiseRecommender(
        n_users=n_users,
        n_items=n_items,
        n_factors=model_params.n_factors,
        lr=model_params.lr,
        n_epochs=model_params.n_epochs,
        reg=model_params.reg,
        pscores=estimated_pscores,
        seed=12345,
        n_positions=k,
    )
    train_user_ids = np.array([[0, 0], [1, 1]])
    train_item_ids = np.array([[0, 1], [2, 3]])
    train_clicks = np.array([[0, 1], [1, 0]])
    train = TrainData(train_user_ids, train_item_ids, train_clicks)

    test_user_ids = np.array([[0, 0], [1, 1]])
    test_item_ids = np.array([[2, 3], [0, 1]])
    test_ratings = np.array([[0.3, 0.7], [0.5, 0.4]])
    test = TestData(test_user_ids, test_item_ids, test_ratings)

    dataset = (train, test)
    train_loss, test_loss, ndcgs = model.fit(dataset)

    assert isinstance(train_loss, list)
    assert isinstance(test_loss, list)
    assert isinstance(ndcgs, list)


def test_listwise_mf():
    n_users = 2
    n_items = 4
    k = 2
    position_bias = (0.9, 1.0)
    estimated_pscores = get_pscores(k, position_bias)
    model_params = ModelConfig()
    model = ListwiseRecommender(
        n_users=n_users,
        n_items=n_items,
        n_factors=model_params.n_factors,
        lr=model_params.lr,
        n_epochs=model_params.n_epochs,
        reg=model_params.reg,
        pscores=estimated_pscores,
        seed=12345,
        n_positions=k,
    )

    train_user_ids = np.array([[0, 0], [1, 1]])
    train_item_ids = np.array([[0, 1], [2, 3]])
    train_clicks = np.array([[0, 1], [1, 0]])
    train = TrainData(train_user_ids, train_item_ids, train_clicks)

    test_user_ids = np.array([[0, 0], [1, 1]])
    test_item_ids = np.array([[2, 3], [0, 1]])
    test_ratings = np.array([[0.3, 0.7], [0.5, 0.4]])
    test = TestData(test_user_ids, test_item_ids, test_ratings)

    dataset = (train, test)
    train_loss, test_loss, ndcgs = model.fit(dataset)

    assert isinstance(train_loss, list)
    assert isinstance(test_loss, list)
    assert isinstance(ndcgs, list)
