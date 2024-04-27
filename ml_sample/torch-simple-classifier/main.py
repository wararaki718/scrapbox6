import torch
from sklearn.datasets import load_iris

from model import NNModel


def main() -> None:
    iris = load_iris()
    X = torch.Tensor(iris.data)
    y = torch.Tensor(iris.target).long()
    print(X.shape)
    print(y.shape)

    n_classes = len(set(iris.target))
    print(n_classes)

    model = NNModel(X.shape[1], n_classes)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    print("model defined")

    model.train()
    epochs = 10
    for epoch in range(1, epochs+1):
        y_preds = model(X)

        optimizer.zero_grad()
        loss: torch.Tensor = criterion(y_preds, y)
        print(f"epoch {epoch:2d}: loss={y_preds[0]}, loss.grad={y_preds.grad}")
        loss.backward()
        optimizer.step()
        
    print("DONE")


if __name__ == "__main__":
    main()
