import torch

from model import Model


def main() -> None:
    X = torch.rand(100, 2)
    y = torch.rand(100)

    model = Model(2, 3, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1} | Loss: {loss.item():.4f}")

    print("DONE")


if __name__ == "__main__":
    main()
