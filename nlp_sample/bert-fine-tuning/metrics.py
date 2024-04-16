import evaluate
import numpy as np


class MetricsCalculator:
    def __init__(self) -> None:
        self._metric = evaluate.load("accuracy")
    
    def compute(self, y_preds: tuple) -> dict:
        logits, labels = y_preds
        predictions = np.argmax(logits, axis=-1)
        return self._metric.compute(predictions=predictions, references=labels)
