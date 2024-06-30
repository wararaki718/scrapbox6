from transformers import AutoTokenizer, AutoModelForTokenClassification

# from postprocessor import Postprocessor
# from vectorizer import TextVectorizer


def main() -> None:
    model_name = "macavaney/deepct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    print("model loaded!")
    print()

    text = "This is a pen."
    tokens = tokenizer(text, return_tensors="pt")
    print(tokens)
    print()

    result = model(**tokens)
    print(result)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
