import optuna
import json

# import numpy as np

# from src.listwiseMF import ListwiseRecommender
from src.pointwise import PointwiseRecommender
from conf.settings.default import DataConfig
from utils.dataloader import synthesize_data, generate_logged_data

params = DataConfig(
    n_users=100,
    n_items=500,
    position_bias=(1.0, 1.0),
    mu_u=1,
    mu_i=100,
    dirichlet_noise=(0.3, 0.01),
    n_factors=10,
    policy="random",
    oracle=False,
    k=5,
    train_test_split=0.8,
)
Vui = synthesize_data(params)
dataset = generate_logged_data(params, Vui)

train = dataset.train[:, :, [0, 1, 3]]
val = dataset.val[:, :, [0, 1, 3]]
test = dataset.test


def objective(trial):
    # ハイパーパラメータの設定
    n_factors = trial.suggest_int("n_factors", 1, 100)
    reg = trial.suggest_float("reg", 1e-5, 1.0, log=True)
    lr = trial.suggest_float("lr", 1e-5, 1.0, log=True)
    scale = trial.suggest_float("scale", 1e-5, 1.0, log=True)
    n_epochs = trial.suggest_int("n_epochs", 1, 100)
    seed = params.seed
    batch_size = trial.suggest_int("batch_size", 50, 200)
    # M = trial.suggest_float("M", 0.01, 0.1)
    # cliped_pscores = np.where(dataset.pscores < M, M, dataset.pscores)

    # モデルの設定
    model = PointwiseRecommender(
        n_users=params.n_users,
        n_items=params.n_items,
        n_factors=n_factors,
        reg=reg,
        lr=lr,
        scale=scale,
        n_epochs=n_epochs,
        seed=seed,
        n_positions=params.k,
        pscores=None,
        batch_size=batch_size,
    )

    # モデルの学習
    val_loss, _, _ = model.fit((train, val, test))

    return val_loss[-1]


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200)
    print(study.best_params)

    dumped_params = json.dumps(study.best_params)

    with open("./data/pointwise_naive_params.json", "w") as f:
        f.write(dumped_params)
