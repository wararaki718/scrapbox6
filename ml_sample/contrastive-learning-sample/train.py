from typing import List

import torch

from criterion import TripletContrastiveLoss
from model import TripletModel
from utils import try_gpu


class Trainer:
    def __init__(self, n_epochs: int=10) -> None:
        self._n_epochs = n_epochs
        self._criterion = TripletContrastiveLoss()


    def train(
        self,
        model: TripletModel,
        optimzier: torch.optim.Optimizer,
        X_train_queries: List[torch.Tensor],
        X_train_positive_documents: List[torch.Tensor],
        X_train_negative_documents: List[torch.Tensor],
        X_valid_queries: List[torch.Tensor],
        X_valid_positive_documents: List[torch.Tensor],
        X_valid_negative_documents: List[torch.Tensor],
    ) -> TripletModel:
        for epoch in range(1, self._n_epochs + 1):
            # model training
            train_loss = self._train_step(
                model, 
                optimzier,
                X_train_queries,
                X_train_positive_documents,
                X_train_negative_documents,
            )
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss}")
            
            # model validation
            valid_loss = self._validate_step(
                model,
                X_valid_queries,
                X_valid_positive_documents,
                X_valid_negative_documents,
            )
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: valid_loss={valid_loss}")

        return model


    def _train_step(
        self,
        model: TripletModel,
        optimizer: torch.optim.Optimizer,
        X_queries: List[torch.Tensor],
        X_positive_documents: List[torch.Tensor],
        X_negative_documents: List[torch.Tensor],
    ) -> float:
        model.train()

        train_loss = 0.0
        for x_queries, x_positive_documents, x_negative_documents in zip(X_queries, X_positive_documents, X_negative_documents):
            # gpu
            x_queries = try_gpu(x_queries)
            x_positive_documents = try_gpu(x_positive_documents)
            x_negative_documents = try_gpu(x_negative_documents)

            # estimate
            (
                query_embeddings,
                positive_document_embeddings,
                negative_document_embeddings,
            ) = model(x_queries, x_positive_documents, x_negative_documents)

            # train
            optimizer.zero_grad()
            loss: torch.Tensor = self._criterion(
                query_embeddings,
                positive_document_embeddings,
                negative_document_embeddings,
            )
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
        
        return train_loss / len(X_queries)


    def _validate_step(
        self,
        model: TripletModel,
        X_queries: List[torch.Tensor],
        X_positive_documents: List[torch.Tensor],
        X_negative_documents: List[torch.Tensor],
    ) -> float:
        model.eval()

        valid_loss = 0.0
        for x_queries, x_positive_documents, x_negative_documents in zip(X_queries, X_positive_documents, X_negative_documents):
            # gpu
            x_queries = try_gpu(x_queries)
            x_positive_documents = try_gpu(x_positive_documents)
            x_negative_documents = try_gpu(x_negative_documents)
    
            # estimate
            (
                query_embeddings,
                positive_document_embeddings,
                negative_document_embeddings,
            ) = model(x_queries, x_positive_documents, x_negative_documents)

            # validation
            loss: torch.Tensor = self._criterion(
                query_embeddings,
                positive_document_embeddings,
                negative_document_embeddings,
            )
            valid_loss += loss.item()
        
        return valid_loss / len(X_queries)
