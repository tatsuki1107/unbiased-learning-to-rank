import numpy as np


def calc_dcg(ratings: np.ndarray, scores: np.ndarray, _k: int = 5) -> float:
    """ Discounted Cumulative Gain (DGC)"""
    test_indices = np.arange(ratings.shape[0])[:, None]
    pred_ratings = ratings[
        test_indices, np.argsort(-scores, axis=1, kind="mergesort")
    ]
    user_relevances = pred_ratings[:, :_k]
    dcgs = user_relevances[:, 0] + np.sum(
        user_relevances[:, 1:]
        / np.log2(np.arange(2, user_relevances.shape[1] + 1) + 1),
        axis=1,
    )
    dcgs = np.where(dcgs != 0, dcgs, np.nan)

    return np.nanmean(dcgs)
