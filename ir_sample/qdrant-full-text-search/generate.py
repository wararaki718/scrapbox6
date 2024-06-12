import random
from typing import List


class VectorGenerator:
    def __init__(self, size: int) -> None:
        self._size = size

    def generate(self) -> List[float]:
        return [random.random() for _ in range(self._size)]
