from typing import List, Dict, Union

from transformers import (
    BertConfig,
    BertJapaneseTokenizer,
    BertForMaskedLM,
    pipeline,
    Pipeline,
)


class MaskEstimator:
    def __init__(
        self,
        config: BertConfig,
        tokenizer: BertJapaneseTokenizer,
        model: BertForMaskedLM,
    ) -> None:
        task = "fill-mask"
        self._estimate: Pipeline = pipeline(
            task,
            model=model,
            tokenizer=tokenizer,
            config=config,
        )

    def estimate(self, text: str) -> List[Dict[str, Union[str, float, int]]]:
        results = self._estimate(text)
        return results
