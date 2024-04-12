from datasets import load_dataset


def main() -> None:
    dataset = load_dataset("unicamp-dl/mmarco", "queries-japanese")
    item = dataset["train"][1]
    print(item)
    print(type(dataset))
    print(len(dataset["train"]))
    print(dataset.keys())
    print("DONE")


if __name__ == "__main__":
    main()
