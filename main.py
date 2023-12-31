from pathlib import Path
import pandas as pd
import joblib
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from conf.settings.default import ExperimentConfig
from utils.dataloader import synthesize_data, generate_logged_data
from src.pointwise import PointwiseRecommender
from src.listwise import ListwiseRecommender


cs = ConfigStore.instance()
cs.store(name="config", node=ExperimentConfig)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """unbiased learning to rankの実験を行う
    (まだ開発途中でこのmain.pyは未完成)

    実験概要: UnbiasedなPointwise損失関数とListwise損失関数の比較を行う

    目的: 推薦の文脈でunbiasedなListwise損失関数が機能し、Pointwise損失関数よりもランク指標が高くなることを確かめたい。

    懸念事項: 論理的にはunbiasedなPointwise, Listwise損失関数は機能するが、完全に
    Implicitなデータしかない場合、biasedな推定量と有意な差が見られない
    """

    data_dir = Path("./data/synthetic.pkl")

    if cfg.dataset.is_created_dataset and cfg.dataset.is_created_clicks:
        with open(data_dir, "rb") as f:
            synthetic_data, dataset = joblib.load(f)

    if cfg.dataset.is_created_dataset and not cfg.dataset.is_created_clicks:
        with open(data_dir, "rb") as f:
            synthetic_data, _ = joblib.load(f)
        dataset = generate_logged_data(cfg.dataset, synthetic_data)

    if not cfg.dataset.is_created_dataset:
        synthetic_data = synthesize_data(cfg.dataset)
        dataset = generate_logged_data(cfg.dataset, synthetic_data)

        with open(data_dir, "wb") as f:
            joblib.dump([synthetic_data, dataset], f)

    dataset = generate_logged_data(cfg.dataset, synthetic_data)

    result_df = pd.DataFrame()

    # implement pointwise model
    pointwise_model = PointwiseRecommender(
        n_users=cfg.dataset.n_users,
        n_items=cfg.dataset.n_items,
        n_factors=cfg.pointwise_config.n_factors,
        lr=cfg.pointwise_config.lr,
        n_epochs=cfg.pointwise_config.n_epochs,
        reg=cfg.pointwise_config.reg,
        seed=cfg.dataset.seed,
        pscores=dataset.pscores,
        n_positions=cfg.dataset.k,
    )

    (
        pointwise_ips_train_loss,
        pointwise_ips_test_loss,
        pointwise_ips_ndcg,
    ) = pointwise_model.fit(dataset)
    result_df["pointwise_ips_train_loss"] = pointwise_ips_train_loss
    result_df["pointwise_ips_test_loss"] = pointwise_ips_test_loss
    result_df["pointwise_ips_ndcg"] = pointwise_ips_ndcg

    pointwise_model = PointwiseRecommender(
        n_users=cfg.dataset.n_users,
        n_items=cfg.dataset.n_items,
        n_factors=cfg.pointwise_config.n_factors,
        lr=cfg.pointwise_config.lr,
        n_epochs=cfg.pointwise_config.n_epochs,
        reg=cfg.pointwise_config.reg,
        seed=cfg.dataset.seed,
        pscores=None,
        n_positions=cfg.dataset.k,
    )

    (
        pointwise_naive_train_loss,
        pointwise_naive_test_loss,
        pointwise_naive_ndcg,
    ) = pointwise_model.fit(dataset)
    result_df["pointwise_naive_train_loss"] = pointwise_naive_train_loss
    result_df["pointwise_naive_test_loss"] = pointwise_naive_test_loss
    result_df["pointwise_naive_ndcg"] = pointwise_naive_ndcg

    # implement listwise model
    listwise_model = ListwiseRecommender(
        n_users=cfg.dataset.n_users,
        n_items=cfg.dataset.n_items,
        n_factors=cfg.listwise_config.n_factors,
        lr=cfg.listwise_config.lr,
        n_epochs=cfg.listwise_config.n_epochs,
        reg=cfg.listwise_config.reg,
        seed=cfg.dataset.seed,
        pscores=dataset.pscores,
        n_positions=cfg.dataset.k,
    )

    (
        listwise_ips_train_loss,
        listwise_ips_test_loss,
        listwise_ips_ndcg,
    ) = listwise_model.fit(dataset)
    result_df["listwise_ips_train_loss"] = listwise_ips_train_loss
    result_df["listwise_ips_test_loss"] = listwise_ips_test_loss
    result_df["listwise_ips_ndcg"] = listwise_ips_ndcg

    listwise_model = ListwiseRecommender(
        n_users=cfg.dataset.n_users,
        n_items=cfg.dataset.n_items,
        n_factors=cfg.listwise_config.n_factors,
        lr=cfg.listwise_config.lr,
        n_epochs=cfg.listwise_config.n_epochs,
        reg=cfg.listwise_config.reg,
        seed=cfg.dataset.seed,
        pscores=None,
        n_positions=cfg.dataset.k,
    )

    (
        listwise_naive_train_loss,
        listwise_naive_test_loss,
        listwise_naive_ndcg,
    ) = listwise_model.fit(dataset)
    result_df["listwise_naive_train_loss"] = listwise_naive_train_loss
    result_df["listwise_naive_test_loss"] = listwise_naive_test_loss
    result_df["listwise_naive_ndcg"] = listwise_naive_ndcg

    result_df.to_csv("./data/result.csv", index=False)


if __name__ == "__main__":
    main()
