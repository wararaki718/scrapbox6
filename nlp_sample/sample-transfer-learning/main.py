from datasets import load_dataset


def main() -> None:
    dataset_name = "dair-ai/emotion"
    dataset = load_dataset(dataset_name)
    print(f"train: {len(dataset['train'])}")
    print(f"valid: {len(dataset['validation'])}")
    print(f"test: {len(dataset['test'])}")

    model_name = "google-bert/bert-base-cased"
    print(type(dataset["train"]))

    cnt = 0
    for x in dataset["train"]:
        cnt += 1
    print(cnt)
    print(x)
    print("DONE")


if __name__ == "__main__":
    main()
