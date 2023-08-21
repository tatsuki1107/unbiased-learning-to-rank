import numpy as np
from dataclasses import dataclass


@dataclass
class Adam:
    param_shape: tuple
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self.M = np.zeros(self.param_shape)
        self.V = np.zeros(self.param_shape)

    def update(self, index: int, grad: np.ndarray) -> np.ndarray:
        self.M[index] = self.beta1 * self.M[index] + (1 - self.beta1) * grad
        self.V[index] = self.beta2 * self.V[index] + (1 - self.beta2) * (
            grad**2
        )
        M_hat = self.M[index] / (1 - self.beta1)
        V_hat = self.V[index] / (1 - self.beta2)
        return M_hat / ((V_hat**0.5) + self.eps)
