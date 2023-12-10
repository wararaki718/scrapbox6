from transformers import BartForConditionalGeneration, BartTokenizer


def main() -> None:
    # model load
    model_name = "facebook/bart-base"
    model = BartForConditionalGeneration.from_pretrained(
        model_name,
        forced_bos_token_id=0
    )
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # generate
    text = "UN Chief Says There Is No <mask> in Syria"
    batch: BartTokenizer = tokenizer(text, return_tensors="pt")
    generated_ids = model.generate(batch["input_ids"])

    result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(result)

    print("DONE")


if __name__ == "__main__":
    main()
