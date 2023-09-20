from qdrant_client.models import VectorParams

from builder import QueryBuilder
from client import SearchClient
from condition import SearchCondition
from utils import get_data, show


def main():
    collection_name = "sample"
    dim = 3

    client = SearchClient()
    params = VectorParams(
        size=dim,
        distance="Cosine"
    )
    _ = client.create_index(collection_name, params)
    print(f"index created: {collection_name}")

    points = get_data()
    client.insert(collection_name, points)
    print(f"data inserted: {len(points.ids)}")
    print()

    print("search:")
    condition = SearchCondition(city="London")
    print(condition)
    query = QueryBuilder.build(condition=condition, vector=[0.9, 0.1, 0.1])
    response = client.search(collection_name, query)
    show(response)

    print("create snapshot:")
    response = client.create_snapshot(collection_name)
    snapshot_name = response.name
    print(f"{response.name}")
    print("")

    print("snapshot list:")
    response = client.list_snapshots(collection_name)
    print(response)
    print()

    print("search:")
    condition = SearchCondition(city="London")
    print(condition)
    query = QueryBuilder.build(condition=condition, vector=[0.9, 0.1, 0.1])
    response = client.search(collection_name, query)
    show(response)

    print("delete snapshot")
    _ = client.delete_snapshot(collection_name, snapshot_name)
    print()

    print("snapshot list:")
    response = client.list_snapshots(collection_name)
    print(response)
    print()

    _ = client.delete_index(collection_name)
    print(f"index deleted: {collection_name}")

    print("DONE")


if __name__ == "__main__":
    main()
