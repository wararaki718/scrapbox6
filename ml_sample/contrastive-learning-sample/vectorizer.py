from functools import partial
from typing import List

import torch
from transformers import BertModel, BertJapaneseTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from utils import try_gpu


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
        self._model = try_gpu(self._model)

    def transform(self, texts: List[str]) -> torch.Tensor:
        inputs: dict = self._tokenize(texts)
        # gpu
        inputs["input_ids"] = try_gpu(inputs["input_ids"])
        inputs["token_type_ids"] = try_gpu(inputs["token_type_ids"])
        inputs["attention_mask"] = try_gpu(inputs["attention_mask"])

        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self._model(**inputs)
        embeddings: torch.Tensor = outputs.last_hidden_state.mean(dim=1).cpu().detach()
        return embeddings
