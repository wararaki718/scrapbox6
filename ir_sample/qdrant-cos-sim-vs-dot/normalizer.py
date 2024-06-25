from typing import List

import numpy as np


class VectorNormalizer:
    def normalize(self, vector: List[float]) -> List[float]:
        x = np.array(vector).reshape(1, -1)
        x_norm: np.ndarray = np.linalg.norm(x, axis=1).reshape(-1, 1)
        norm_vector: List[float] = (x / x_norm).flatten().tolist()

        return norm_vector
