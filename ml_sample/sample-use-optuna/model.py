import torch
import torch.nn as nn
from optuna.trial import Trial


class NNModel(nn.Module):
    def __init__(self, trial: Trial, n_input: int=28*28, n_output: int=10) -> None:
        super(NNModel, self).__init__()

        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = []
        for n_layer in range(n_layers):
            n_hidden = trial.suggest_int(f"n_hidden_layer_{n_layer}", 16, 64)
            layers.append(nn.Linear(n_input, n_hidden))
            layers.append(nn.ReLU())
            
            p = trial.suggest_float(f"dropout_layer_{n_layer}", 0.2, 0.5)
            layers.append(nn.Dropout(p))
            n_input = n_hidden
        layers.append(nn.Linear(n_input, n_output))
        layers.append(nn.LogSoftmax(dim=1))

        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
