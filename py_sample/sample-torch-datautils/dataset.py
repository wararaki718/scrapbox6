from typing import Generator, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset


class MapStyleDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self._X = X
        self._y = y

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._X[index]
        y = self._y[index]
        return x, y
    
    def __len__(self) -> int:
        return len(self._X)


class InterableStyleDataset(IterableDataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self._X = X
        self._y = y

    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        for x, y in zip(self._X, self._y):
            yield (x, y)
