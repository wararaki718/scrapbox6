from typing import Generator, List


class DataIterator:
    def __init__(self, X: List[str], chunksize: int=256) -> None:
        self._X = X
        self._chunksize = chunksize

    def __iter__(self) -> Generator[List[str], None, None]:
        for i in range(0, len(self._X), self._chunksize):
            x_chunk = self._X[i: i+self._chunksize]
            yield x_chunk
