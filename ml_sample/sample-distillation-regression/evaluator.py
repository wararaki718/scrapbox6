import torch
from torch.utils.data import DataLoader

from utils import try_gpu


class Evaluator:
    def evaluate(self, model: torch.nn.Module, loader: DataLoader) -> float:
        model.eval()

        correct = 0.0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = try_gpu(inputs)
                labels = try_gpu(labels)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total
        return accuracy


class DistilEvaluator:
    def evaluate(self, model: torch.nn.Module, loader: DataLoader) -> float:
        model.eval()

        correct = 0.0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = try_gpu(inputs)
                labels = try_gpu(labels)

                outputs, _ = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total
        return accuracy
