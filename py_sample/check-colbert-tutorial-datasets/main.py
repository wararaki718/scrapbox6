import gc
from datasets import load_dataset


def main() -> None:
    dataset_name = "lifestyle"
    passages = load_dataset("colbertv2/lotte_passages", dataset_name)

    print("check passages:")
    print(type(passages))
    print(passages.keys())
    print(type(passages["dev_collection"]))
    print(passages.column_names)
    print(passages.shape)
    print()

    dev_collections = passages["dev_collection"]
    print(type(dev_collections["doc_id"]))
    print(type(dev_collections["author"]))
    print(type(dev_collections["text"]))
    print()

    print(dev_collections["doc_id"][:3])
    print(dev_collections["author"][:3])
    print(dev_collections["text"][:3])
    print()

    del dev_collections, passages
    gc.collect()

    print("queries:")
    queries = load_dataset("colbertv2/lotte", dataset_name)
    print(type(queries))
    print(queries.keys())
    print(type(queries["search_dev"]))
    print(queries.column_names)
    print(queries.shape)

    dev_queries = queries["search_dev"]
    print(type(dev_queries["qid"]))
    print(type(dev_queries["query"]))
    print(type(dev_queries["author"]))
    print(type(dev_queries["answers"]))
    print()

    print(dev_queries["qid"][:3])
    print(dev_queries["query"][:3])
    print(dev_queries["author"][:3])
    print(dev_queries["answers"][:3])
    print()

    del dev_queries, queries
    gc.collect()

    print("DONE")


if __name__ == "__main__":
    main()
