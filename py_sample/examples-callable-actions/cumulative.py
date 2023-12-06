# stateful callables

class CumulativeAverager:
    def __init__(self) -> None:
        self._data = list()
    
    def __call__(self, x: int) -> float:
        self._data.append(x)
        return sum(self._data) / len(self._data)
