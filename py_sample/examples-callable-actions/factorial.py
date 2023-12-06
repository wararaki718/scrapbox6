# caching

class Factorial:
    def __init__(self) -> None:
        self._cache = {0: 1, 1: 1}

    def __call__(self, x: int) -> int:
        if x not in self._cache:
            self._cache[x] = x * self(x - 1)
        return self._cache[x]
