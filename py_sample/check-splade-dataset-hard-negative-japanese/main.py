from datasets import load_dataset


def main() -> None:
    dataset_name = "bclavie/mmarco-japanese-hard-negatives"
    data = load_dataset(dataset_name)
    print(data.shape)
    print(data.column_names)
    print(type(data))
    print(data)
    print("DONE")


if __name__ == "__main__":
    main()
