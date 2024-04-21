from functools import partial
from typing import List

import pandas as pd
import torch
from tqdm import tqdm
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
    
    def transform(self, texts: pd.Series, chunksize: int=32) -> List[torch.Tensor]:
        X = []
        for i in tqdm(range(0, len(texts), chunksize)):
            # tokenize
            texts_chunk: List[str] = texts.iloc[i: i+chunksize].tolist()
            inputs: BatchEncoding = self._tokenize(texts_chunk)

            # apply gpu
            inputs["input_ids"] = try_gpu(inputs["input_ids"])
            inputs["token_type_ids"] = try_gpu(inputs["token_type_ids"])
            inputs["attention_mask"] = try_gpu(inputs["attention_mask"])

            # vectorize
            outputs = self._model(**inputs)
            embeddings: torch.Tensor = outputs.last_hidden_state[:, 0, :]

            X.append(embeddings.cpu().detach())
        
        return X


class LabelVectorizer:
    def transform(self, labels: pd.Series) -> torch.Tensor:
        return torch.Tensor(labels).long()
