from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class DenseVectorizer:
    def __init__(self, model_name: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)

    def transform(self, text: str) -> List[float]:
        inputs: dict = self._tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
        outputs = self._model(**inputs)
        embeddings: torch.Tensor = outputs.last_hidden_state[0].mean(axis=0)
        return embeddings.cpu().detach().tolist()


class RandomVectorizer:
    def __init__(self, size: int=768) -> None:
        self._size = size

    def transform(self) -> List[float]:
        x: np.ndarray = np.random.random(self._size)
        return x.tolist()
