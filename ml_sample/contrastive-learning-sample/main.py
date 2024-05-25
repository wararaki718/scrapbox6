from datasets import load_dataset


def main() -> None:
    # use
    dataset_name = "bclavie/mmarco-japanese-hard-negatives"
    data = load_dataset(dataset_name)
    print(data.shape)
    print(data.column_names)
    print(type(data))
    print(data)
    print(type(data["train"]["query"]))
    print(type(data["train"]["positives"]))
    print(type(data["train"]["negatives"]))

    # TODO remove
    # interaction_name = "unicamp-dl/mmarco"
    # interaction = load_dataset(interaction_name, "japanese")
    # print(interaction.shape)
    # print(interaction.column_names)

    # print(interaction["train"]["query"][:3])
    # print()

    # print(interaction["train"]["positive"][:3])
    # print()

    # print(interaction["train"]["negative"][:3])
    # print()

    print("DONE")


if __name__ == "__main__":
    main()
