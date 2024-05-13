from argparse import ArgumentParser, Namespace

from client import SearchClient


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--collection-name", default="newsgroups")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    collection_name: str = args.collection_name
    print(f"collection name: {collection_name}")

    client = SearchClient()

    # delete index
    _ = client.delete_index(collection_name)
    print(f"collection deleted: {collection_name}")
    print("DONE")


if __name__ =="__main__":
    main()
