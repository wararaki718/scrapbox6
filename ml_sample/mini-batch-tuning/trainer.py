from typing import List

import torch
from torch.utils.data import DataLoader

from model import NNModel
from utils import try_gpu


class Trainer:
    def __init__(self, n_epochs: int=10) -> None:
        self._n_epochs = n_epochs
        self._criterion = torch.nn.CrossEntropyLoss()

    def train(
        self,
        model: NNModel,
        optimzier: torch.optim.Optimizer,
        train_loader: DataLoader,
        valid_loader: DataLoader,
    ) -> NNModel:
        for epoch in range(1, self._n_epochs + 1):
            # model training
            train_loss = self._train_step(
                model, 
                optimzier,
                train_loader,
            )
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss}")
            
            # model validation
            valid_loss = self._validate_step(
                model,
                valid_loader,
            )
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: valid_loss={valid_loss}")

        return model

    def _train_step(
        self,
        model: NNModel,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
    ) -> float:
        model.train()

        train_loss = 0.0
        for X, y in train_loader:
            # gpu
            X: torch.Tensor = try_gpu(X)
            y: torch.Tensor = try_gpu(y)

            # estimate
            y_pred = model(X)

            # train
            optimizer.zero_grad()
            loss: torch.Tensor = self._criterion(y_pred, y)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
        
        return train_loss / train_loader.batch_size

    def _validate_step(
        self,
        model: NNModel,
        valid_loader: DataLoader,
    ) -> float:
        model.eval()

        valid_loss = 0.0
        for X, y in valid_loader:
            # gpu
            X: torch.Tensor = try_gpu(X)
            y: torch.Tensor = try_gpu(y)
    
            # estimate
            y_pred = model(X)

            # validation
            loss: torch.Tensor = self._criterion(y_pred, y)
            valid_loss += loss.item()
        
        return valid_loss / valid_loader.batch_size
