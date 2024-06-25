from typing import List

import numpy as np


class RandomVectorizer:
    def __init__(self, size: int=768) -> None:
        self._size = size

    def generate(self) -> List[float]:
        x: np.ndarray = np.random.random(self._size)
        return x.tolist()
