import torch
import torch.nn as nn


class NNModel(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int=128) -> None:
        super(NNModel, self).__init__()
        layers = [
            nn.Linear(n_input, n_hidden),
            nn.Dropout(p=0.9),
            nn.Sigmoid(),
            nn.Linear(n_hidden, n_hidden),
            nn.Dropout(p=0.9),
            nn.Sigmoid(),
            nn.Linear(n_hidden, n_output),
        ]
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)
