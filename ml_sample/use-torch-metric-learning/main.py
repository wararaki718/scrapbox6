import torch
from pytorch_metric_learning import losses, distances, miners


def main() -> None:
    n = 20
    X = torch.randn(n, 128)
    y = torch.randint(0, 2, (n,))
    print("X, y:")
    print(X)
    print(y)
    print()

    miner = miners.TripletMarginMiner()
    data = miner(X, y)
    print("miner output:")
    print(data)
    print()

    criterion = losses.TripletMarginLoss(distance=distances.CosineSimilarity())
    loss = criterion(X, y, data)
    print("loss:")
    print(loss)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
