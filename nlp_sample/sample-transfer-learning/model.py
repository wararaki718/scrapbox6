import torch


class NNModel(torch.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int
    ) -> None:
        super().__init__()

        layers = [
            torch.nn.Linear(n_input, n_hidden),
            torch.nn.Dropout(p=0.001),
            torch.nn.Sigmoid(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.Dropout(p=0.001),
            torch.nn.Sigmoid(),
            torch.nn.Linear(n_hidden, n_output),
            torch.nn.Softmax(dim=1),
        ]
        self._model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)
