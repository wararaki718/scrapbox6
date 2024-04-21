from typing import List

import torch

from train import Trainer
from valid import Validator


class TrainValidExecuter:
    def __init__(self) -> None:
        self._trainer = Trainer()
        self._validator = Validator()

    def execute(
        self,
        model: torch.nn.Module,
        X_train: List[torch.Tensor],
        y_train: torch.Tensor,
        X_valid: List[torch.Tensor],
        y_valid: torch.Tensor,
        epochs: int = 100,
    ) -> float:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        total_loss = 0.0
        for epoch in range(1, epochs+1):
            train_loss = self._trainer.train(model, X_train, y_train, criterion, optimizer)
            valid_loss = self._validator.validate(model, X_valid, y_valid, criterion)
            total_loss = valid_loss
            if epoch % 10 == 0:
                print(f"epoch {epoch}: train_loss={train_loss}, valid_loss={valid_loss}")
        
        return total_loss / epoch
