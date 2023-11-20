from random import random
from typing import Tuple

import numpy as np


class PSOSimulator:
    def __init__(self, w: float=0.5, C1: float=0.05, C2: float=0.01) -> None:
        self._w = w
        self._C1 = C1
        self._C2 = C2

    def simulate(self, X: np.ndarray, V: np.ndarray, t: int=100) -> None:
        self._pbest = X.copy() # (n_particle, n_dim)
        self._gbest = X[0] # (n_dim,)
        self._evaluate(X)

        for i in range(1, t+1):
            self._evaluate(X)
            X, V = self._update(X, V)
            print(f"{i}-th best: {self._fitness(self._gbest)} {self._gbest}")
            # todo: early stopping
    
    def _fitness(self, x: np.ndarray) -> float:
        return 1.0 - (x.mean() ** 2)

    def _evaluate(self, X: np.ndarray) -> None:
        g_score = self._fitness(self._gbest)
        for i, x_i in enumerate(X):
            x_score = self._fitness(x_i)
            p_score = self._fitness(self._pbest[i])
            if x_score > p_score:
                self._pbest[i] = x_i
            
            if x_score > g_score:
                self._gbest = x_i

    def _update(self, X: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        V_next = self._w * V + self._C1 * random() * (self._pbest - X) + self._C2 * random() * (self._gbest - X)
        X_next = X + V_next
        return X_next, V_next
