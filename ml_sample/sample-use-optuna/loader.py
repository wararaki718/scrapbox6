from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DataLoaderFactory:
    @classmethod
    def create(cls, filename: Path, batch_size: int=128) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            datasets.FashionMNIST(filename, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=batch_size,
            shuffle=True,
        )

        valid_loader = DataLoader(
            datasets.FashionMNIST(filename, train=False, transform=transforms.ToTensor()),
            batch_size=batch_size,
            shuffle=True,
        )
        
        return train_loader, valid_loader
