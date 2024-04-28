import torch
import torch.nn as nn


class NNModel(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int=8) -> None:
        super(NNModel, self).__init__()

        self._model = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Sigmoid(),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._model(x)
        return x
