import random
from typing import List


class ConditionGenerator:
    def __init__(self, size: int) -> None:
        self._size = size

    def generate(self) -> List[str]:
        x = []
        for i in range(1, self._size+1):
            tmp = random.randint(0, 1)
            if tmp > 0:
                label = f"color_{i}"
                x.append(label)
        return x


class VectorGenerator:
    def __init__(self, size: int) -> None:
        self._size = size

    def generate(self) -> List[float]:
        return [random.random() for _ in range(self._size)]
