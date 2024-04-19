from functools import partial
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from utils import try_gpu


class TextVectorizer:
    def __init__(self, model_name: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cuda")
        self._tokenize = partial(
            tokenizer.batch_encode_plus,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )
        self._model = AutoModel.from_pretrained(model_name, device_map="cuda")
    
    def transform(self, texts: List[str]) -> torch.Tensor:
        inputs: BatchEncoding = self._tokenize(texts)
        inputs["input_ids"] = try_gpu(inputs["input_ids"])
        inputs["token_type_ids"] = try_gpu(inputs["token_type_ids"])
        inputs["attention_mask"] = try_gpu(inputs["attention_mask"])
        outputs = self._model(**inputs)
        embeddings: torch.Tensor = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu().detach()
