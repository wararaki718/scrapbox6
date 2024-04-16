from functools import partial

import torch
from transformers import AutoTokenizer


class TextTokenizer:
    def __init__(self, model_name: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenize = partial(tokenizer, padding="max_length", truncation=True)
    
    def tokenize(self, x: dict) -> torch.Tensor:
        return self._tokenize(x["text"])
