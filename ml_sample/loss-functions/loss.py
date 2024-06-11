import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x) # prevent overflow
    return np.exp(x) / np.sum(np.exp(x))


def _cross_entropy(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -np.sum(np.log(y[np.arange(y.shape[0]), t] + 1e-7)) / y.shape[0]


def softmax_loss(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    y = _softmax(x)
    loss = _cross_entropy(y, t)
    return loss


def norm_softmax_loss(x: np.ndarray, t: np.ndarray, lambda_: float) -> np.ndarray:
    y = _softmax(lambda_ * x)
    loss = _cross_entropy(y, t)
    return loss
