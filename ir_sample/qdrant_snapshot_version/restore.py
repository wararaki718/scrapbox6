from qdrant_client.models import VectorParams

from client import SearchClient


def main():
    collection_name = "sample"
    snapshot_name = "./download/sample-1.5.1.snapshot"
    dim = 3

    client = SearchClient()
    params = VectorParams(
        size=dim,
        distance="Cosine"
    )
    _ = client.create_index(collection_name, params)
    print(f"index created: {collection_name}")

    _ = client.restore_snapshot(collection_name, snapshot_name)
    print("restored")

    print("DONE")


if __name__ == "__main__":
    main()
