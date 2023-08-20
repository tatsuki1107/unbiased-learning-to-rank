import numpy as np
from utils.models import LogDataset
from conf.settings.default import DataConfig
from utils.dataloader import synthesize_data, generate_logged_data


def test_synthesize_data():
    params = DataConfig()
    Vui = synthesize_data(params)

    assert isinstance(Vui, np.ndarray)
    assert Vui.shape == (params.n_users, params.n_items)
    assert (Vui >= 0).all() and (Vui <= 1).all()


def test_generate_logged_data():
    params = DataConfig()
    Vui = synthesize_data(params)

    dataset = generate_logged_data(params, Vui)

    assert isinstance(dataset, LogDataset)
    assert isinstance(dataset.train, np.ndarray)
    assert isinstance(dataset.val, np.ndarray)
    assert isinstance(dataset.test, np.ndarray)
    assert isinstance(dataset.pscores, np.ndarray)
    assert isinstance(dataset.logged_data_matrix, np.ndarray)
    assert dataset.logged_data_matrix.shape == (
        params.n_users,
        params.n_items,
    )
