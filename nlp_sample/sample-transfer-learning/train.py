from typing import List

import torch

from utils import try_gpu


class Trainer:
    def __init__(self) -> None:
        pass

    def train(
        self,
        model: torch.nn.Module,
        X_train: List[torch.Tensor],
        y_train: torch.Tensor,
        epochs: int = 100,
    ) -> float:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        total_loss = 0.0
        for epoch in range(1, epochs+1):
            train_loss = 0.0
            for i, X in enumerate(X_train):
                X = try_gpu(X)

                y = y_train[i*X.shape[0]: i*X.shape[0]+X.shape[0]]
                y = try_gpu(y)

                optimizer.zero_grad()
                y_preds: torch.Tensor = model(X)
                loss: torch.Tensor = criterion(y_preds, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"epoch {epoch}: {train_loss}")
            
            total_loss += train_loss
        
        return total_loss / epoch
