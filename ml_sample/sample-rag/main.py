from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration


def main() -> None:
    model_name = "facebook/rag-token-nq"
    tokenizer = RagTokenizer.from_pretrained(model_name)
    retriever = RagRetriever.from_pretrained(model_name, index_name="exact", use_dummy_dateset=True)
    model = RagTokenForGeneration.from_pretrained(model_name, retriever=retriever)

    print(type(model))
    text = "who holds the record in 100m freestyle"
    input_dict = tokenizer.prepare_seq2seq_batch(text, return_tensors="pt") 

    generated = model.generate(input_ids=input_dict["input_ids"])
    tokens = tokenizer.batch_decode(generated, skip_special_tokens=True)
    print(tokens[0])
    print("DONE")


if __name__ == "__main__":
    main()
