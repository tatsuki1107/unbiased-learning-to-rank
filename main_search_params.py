import optuna
import json

# from src.listwise import ListwiseRecommender

from src.pointwise import PointwiseRecommender
from conf.settings.default import DataConfig
from utils.dataloader import synthesize_data, generate_logged_data

params = DataConfig()
Vui = synthesize_data(params)
dataset = generate_logged_data(params, Vui)

train = dataset.train[:, :, [0, 1, 2]]
val = dataset.val[:, :, [0, 1, 2]]
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

    with open("./data/pointwise_params.json", "w") as f:
        f.write(dumped_params)
