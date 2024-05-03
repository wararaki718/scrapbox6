import gc

from datasets import load_dataset


def main() -> None:
    query_dataset = load_dataset("unicamp-dl/mmarco", "queries-japanese")
    print(type(query_dataset))
    print(query_dataset.keys())
    print()

    print("train:")
    query_train = query_dataset["train"]
    print(type(query_train))
    print(query_train.column_names)
    print(query_train.shape)
    print()

    print("dev:")
    query_dev = query_dataset["dev"]
    print(query_dev)
    print(query_dev.column_names)
    print(query_dev.shape)
    print()

    print("dev.full:")
    query_dev_full = query_dataset["dev.full"]
    print(query_dev_full)
    print(query_dev_full.column_names)
    print(query_dev_full.shape)
    print()
    del query_dev, query_dev_full, query_train, query_dataset
    gc.collect()

    collection_dataset = load_dataset("unicamp-dl/mmarco", "collection-japanese")
    print(type(collection_dataset))
    print(collection_dataset.keys())
    print()

    print("collection:")
    collection = collection_dataset["collection"]
    print(collection)
    print(collection.column_names)
    print(collection.shape)
    print()

    del collection, collection_dataset
    gc.collect()

    print("DONE")


if __name__ == "__main__":
    main()
