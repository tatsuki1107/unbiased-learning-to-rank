import numpy as np
from utils.models import TrainData, TestData
from conf.settings.default import DataConfig
from utils.dataloader import synthesize_data, generate_clicks


def test_synthesize_data():
    params = DataConfig()
    Vui = synthesize_data(params)

    assert isinstance(Vui, np.ndarray)
    assert Vui.shape == (params.n_users, params.n_items)
    assert (Vui >= 0).all() and (Vui <= 1).all()


def test_generate_clicks():
    params = DataConfig()
    Vui = synthesize_data(params)

    traindata, testdata = generate_clicks(params, Vui)

    assert isinstance(traindata, TrainData)
    assert isinstance(testdata, TestData)
