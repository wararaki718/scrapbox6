from fairseq.models.bart import BARTModel


def main() -> None:
    # load
    model = BARTModel.from_pretrained("./bart.large.mnli", checkpoint_file="model.pt")
    model.eval()

    text1 = "BART is a seq2seq model."
    text2 = "BART is not sequence to sequence."

    tokens = model.encode(text1, text2)
    print(tokens)
    print()

    pred = model.predict("mnli", tokens)
    print(pred)
    print()

    result = pred.argmax()
    print(result) # 0: contradiction, 1: neutral, 2: entialment
    print()

    print("DONE")


if __name__ == "__main__":
    main()
