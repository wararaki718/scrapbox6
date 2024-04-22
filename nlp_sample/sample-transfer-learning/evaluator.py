from typing import Dict, List

import torch
from torcheval.metrics.functional import multiclass_accuracy, multiclass_auroc, multiclass_f1_score

from utils import try_gpu


class Evaluator:
    def evaluate(
        self,
        model: torch.nn.Module,
        X_test: List[torch.Tensor],
        y_test: torch.Tensor,
        n_classes: int,
    ) -> Dict[str, float]:
        model.eval()

        y_preds = []
        with torch.no_grad():
            for i, X in enumerate(X_test):
                X = try_gpu(X)

                y = y_test[i*X.shape[0]: i*X.shape[0]+X.shape[0]]
                y = try_gpu(y)

                y_pred: torch.Tensor = model(X)
                y_preds.append(y_pred.cpu().detach())
        
        y_pred_all = torch.concat(y_preds)
        results = {
            "accuracy": multiclass_accuracy(y_pred_all, y_test),
            "auroc": multiclass_auroc(y_pred_all, y_test, num_classes=n_classes),
            "f1_score": multiclass_f1_score(y_pred_all, y_test),
        }
        return results
