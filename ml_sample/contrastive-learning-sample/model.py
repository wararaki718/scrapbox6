from typing import Tuple

import torch


class NNModel(torch.nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int=256) -> None:
        super(NNModel, self).__init__()
        layers = [
            torch.nn.Linear(n_input, n_hidden),
            torch.nn.Dropout(p=0.9),
            torch.nn.Sigmoid(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.Dropout(p=0.9),
            torch.nn.Sigmoid(),
            torch.nn.Linear(n_hidden, n_output),
        ]
        self._model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class TripletModel(torch.nn.Module):
    def __init__(
        self,
        n_query_input: int,
        n_document_input: int,
        n_query_output: int,
        n_document_output: int,
    ) -> None:
        super(TripletModel, self).__init__()
        self._query_model = NNModel(n_query_input, n_query_output)
        self._document_model = NNModel(n_document_input, n_document_output)
    
    def forward(
        self,
        x_query: torch.Tensor,
        x_positive_document: torch.Tensor,
        x_negative_document: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_embeddings = self._query_model(x_query)
        positive_document_embeddings = self._document_model(x_positive_document)
        negative_document_embeddings = self._document_model(x_negative_document)
        return query_embeddings, positive_document_embeddings, negative_document_embeddings
