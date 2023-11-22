import torch


class Model(torch.nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int) -> None:
        super(Model, self).__init__()
        self._layers = torch.nn.ModuleList([
            torch.nn.Linear(n_input, n_hidden),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(n_hidden, n_output)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            x = layer(x)
        return x
