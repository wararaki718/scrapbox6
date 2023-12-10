from fairseq.models.bart import BARTModel


def main() -> None:
    # load
    model = BARTModel.from_pretrained("./bart.base", checkpoint_file="model.pt")
    model.eval()

    text = "hello world!"
    tokens = model.encode(text)
    print(f"encode: '{text}'")
    print(f"tokens: {tokens.tolist()}")
    result = model.decode(tokens)
    print(f"decode: {result}")
    print()

    features = model.extract_features(tokens)
    print("last layer embeddings:")
    print(f"features: {features.size()}")
    print(f"features: {features}")
    print()

    all_features = model.extract_features(tokens)
    print("all layers embeddings:")
    print(f"features: {all_features.size()}")
    print()

    print("DONE")


if __name__ == "__main__":
    main()
