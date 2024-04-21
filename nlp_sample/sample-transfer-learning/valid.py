from typing import List

import torch

from utils import try_gpu


class Validator:
    def validate(
        self,
        model: torch.nn.Module,
        X_valid: List[torch.Tensor],
        y_valid: torch.Tensor,
        criterion: torch.nn.CrossEntropyLoss,
    ) -> float:
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for i, X in enumerate(X_valid):
                X = try_gpu(X)

                y = y_valid[i*X.shape[0]: i*X.shape[0]+X.shape[0]]
                y = try_gpu(y)

                y_preds: torch.Tensor = model(X)

                loss: torch.Tensor = criterion(y_preds, y)
                total_loss += loss.item()
        return total_loss
