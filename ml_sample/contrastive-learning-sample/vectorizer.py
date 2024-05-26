from functools import partial
from typing import List

import torch
from transformers import BertModel, BertJapaneseTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class DenseVectorizer:
    def __init__(self, model_name: str, max_length: int=512) -> None:
        tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
        self._tokenize = partial(
            tokenizer.batch_encode_plus,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        self._model = BertModel.from_pretrained(model_name)

    def transform(self, texts: List[str]) -> torch.Tensor:
        inputs: dict = self._tokenize(texts)
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self._model(**inputs)
        embeddings: torch.Tensor = outputs.last_hidden_state.mean(dim=1)
        return embeddings
