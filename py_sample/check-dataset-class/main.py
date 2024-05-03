from datasets import Dataset


def main() -> None:
    data = {
        "id": [1, 2, 3],
        "name": ["a", "b", "c"],
    }
    dataset = Dataset.from_dict(data)
    print(dataset)
    print(dataset["id"])
    print(dataset["name"])

    dataset.to_csv("data/output.csv")
    print("DONE")


if __name__ == "__main__":
    main()
