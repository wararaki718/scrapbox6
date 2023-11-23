import numpy as np


class BaseFitness:
    def calculate(self, x: np.ndarray) -> float:
        raise NotImplementedError
    
    def evaluate(self, y: float) -> float:
        raise NotImplementedError


class AckleyFunctionFitness(BaseFitness):
    def calculate(self, x: np.ndarray) -> float:
        return 20.0 - 20.0 * np.exp(-0.2 * np.sqrt(np.power(x, 2).sum()) / len(x)) + np.e - np.exp(np.cos(2 * np.pi * x).sum() / len(x))

    def evaluate(self, x: float) -> float:
        return np.abs(0.0 - self.calculate(x))
