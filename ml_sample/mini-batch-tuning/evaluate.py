import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Evaluator:
    def evaluate(self, model: nn.Module, loader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in loader:
                y_pred = model(X)
                _, y_labels = torch.max(y_pred, 1)
                total += y.size(0)
                correct += (y_labels == y).sum().item()
        return correct / total
