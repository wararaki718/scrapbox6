from typing import List

import torch
import torch.nn.functional as F

from model import TripletModel
from utils import try_gpu


class TripletEvaluator:
    def evaluate(
        self,
        model: TripletModel,
        X_queries: List[torch.Tensor],
        X_positive_documents: List[torch.Tensor],
        X_negative_documents: List[torch.Tensor],
    ) -> float:
        model.eval()

        n_total = 0
        n_true = 0
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

            positive_similarities = F.cosine_similarity(query_embeddings, positive_document_embeddings)
            negative_similarities = F.cosine_similarity(query_embeddings, negative_document_embeddings)

            n_true += (positive_similarities > negative_similarities).sum().item()
            n_total += len(positive_document_embeddings)

        # accuracy
        return n_true / n_total
