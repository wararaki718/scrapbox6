from transformers import BertConfig, BertJapaneseTokenizer, BertForMaskedLM

from estimator import MaskEstimator
from utils import show


def main() -> None:
    model_name = "tohoku-nlp/bert-base-japanese-whole-word-masking"
    
    config = BertConfig.from_pretrained(model_name)
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    estimator = MaskEstimator(config, tokenizer, model)

    text = "私は、[MASK]を食べるのが楽しみだ。"
    results = estimator.estimate(text)
    print()

    print(f"base: {text}")
    print()
    show(results)

    print("DONE")


if __name__ == "__main__":
    main()
