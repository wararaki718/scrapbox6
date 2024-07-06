import torch
from torch.utils.data import DataLoader

from dataset import MapStyleDataset, InterableStyleDataset


def main() -> None:
    n_data = 30
    dim = 4
    batch_size = 5

    X = torch.Tensor([[i for _ in range(dim)] for i in range(n_data)])
    y = torch.Tensor([i for i in range(n_data)])
    print("dataset:")
    print(X.shape)
    print(y.shape)
    print()

    # map-style datasets
    print("map-style:")
    dataset = MapStyleDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size)
    for x_, y_ in loader:
        print(x_, y_)
    print()

    # iterable-style datasets
    print("iterable-style:")
    dataset = InterableStyleDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size)
    for x_, y_ in loader:
        print(x_, y_)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
