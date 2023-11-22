import torch


class SequentialFactory:
    @classmethod
    def create(cls, n_input: int, n_hidden: int, n_output: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(n_input, n_hidden),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(n_hidden, n_output)
        )
