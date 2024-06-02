from functools import partial
from typing import Generator, List

import torch
from tqdm import tqdm
from transformers import BertModel, BertJapaneseTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from utils import try_gpu


class DataIterator:
    def __init__(self, X: List[str], chunksize: int=256) -> None:
        self._X = X
        self._chunksize = chunksize

    def __iter__(self) -> Generator[List[str], None, None]:
        for i in range(0, len(self._X), self._chunksize):
            x_chunk = self._X[i: i+self._chunksize]
            yield x_chunk


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

    def transform(self, texts: List[str], chunksize: int=256) -> List[torch.Tensor]:
        embeddings = []
        data_iterator = DataIterator(texts, chunksize)
        for chunk_texts in tqdm(data_iterator):
            inputs: dict = self._tokenize(chunk_texts)
            # gpu
            inputs["input_ids"] = try_gpu(inputs["input_ids"])
            inputs["token_type_ids"] = try_gpu(inputs["token_type_ids"])
            inputs["attention_mask"] = try_gpu(inputs["attention_mask"])

            outputs: BaseModelOutputWithPoolingAndCrossAttentions = self._model(**inputs)
            chunk_embeddings: torch.Tensor = outputs.last_hidden_state.mean(dim=1).cpu().detach()
            embeddings.append(chunk_embeddings)
        return embeddings
