from sentence_transformers import CrossEncoder


def main() -> None:
    model_name = "tohoku-nlp/bert-base-japanese-v3"
    model = CrossEncoder(model_name, max_length=512)

    scores = model.predict([('Query', 'Paragraph1'), ('Query', 'Paragraph2') , ('Query', 'Paragraph3')])
    print(scores)
    print("DONE")


if __name__ == "__main__":
    main()
