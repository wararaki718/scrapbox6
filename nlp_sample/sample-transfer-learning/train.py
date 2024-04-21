from typing import List

import torch

from utils import try_gpu


class Trainer:
    def train(
        self,
        model: torch.nn.Module,
        X_train: List[torch.Tensor],
        y_train: torch.Tensor,
        criterion: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        model.train()
        total_loss = 0.0
        for i, X in enumerate(X_train):
            X = try_gpu(X)

            y = y_train[i*X.shape[0]: i*X.shape[0]+X.shape[0]]
            y = try_gpu(y)

            optimizer.zero_grad()
            y_preds: torch.Tensor = model(X)
            loss: torch.Tensor = criterion(y_preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        return total_loss
