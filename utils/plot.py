import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(matrix: np.ndarray) -> None:
    """評価値matrixをヒートマップで可視化
    args:
        matrix: 評価値行列
    """
    fig, ax = plt.subplots(figsize=(20, 5))

    my_cmap = plt.cm.get_cmap("Reds")
    heatmap = plt.pcolormesh(matrix.T, cmap=my_cmap)
    plt.colorbar(heatmap)
    ax.grid()
    plt.tight_layout()
    plt.show()
