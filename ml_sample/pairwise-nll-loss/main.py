import torch
import torch.nn as nn

class SoftmaxModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self._softmax = nn.LogSoftmax(dim=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self._softmax(X)


def main() -> None:
    X = torch.randn(3, 5, requires_grad=True)
    y = torch.tensor([1, 0, 4])

    model = SoftmaxModel()
    model.train()
    y_pred = model(X)
    print(y_pred)
    print()

    nll_loss = nn.NLLLoss()
    loss = nll_loss(y_pred, y)
    loss.backward()
    print(loss)
    print("DONE")


if __name__ == "__main__":
    main()
