import time
from qdrant_client.models import VectorParams

from builder import QueryBuilder, QueryMatchAnyBuilder, QueryTextBuilder
from client import SearchClient
from condition import SearchCondition
from generate import VectorGenerator
from utils import get_data, show


def main():
    collection_name = "sample"
    dim = 128

    client = SearchClient()
    params = VectorParams(
        size=dim,
        distance="Cosine",
    )
    _ = client.create_index(collection_name, params)
    print(f"index created: {collection_name}")

    # data
    points = get_data(n_vector=dim)
    _ = client.insert(collection_name, points)
    _ = client.create_payload_index(collection_name, "color")
    print(f"data inserted")
    print()

    # vectorizer
    vector_generator = VectorGenerator(dim)
    vector = vector_generator.generate()

    # search
    condition = SearchCondition(color=["red"])
    query = QueryMatchAnyBuilder.build(
        condition=condition,
        vector=vector,
    )
    start_tm = time.time()
    response = client.search(collection_name, query)
    search_tm = time.time() - start_tm
    print("## QueryMatchAnyBuilder")
    print(f"'matchany condition' search time: {search_tm}")
    print(condition)
    print(query.query_filter)
    show(response)
    print()

    condition = SearchCondition(color=["red", "blue"])
    query = QueryMatchAnyBuilder.build(condition=condition, vector=vector)
    start_tm = time.time()
    response = client.search(collection_name, query)
    search_tm = time.time() - start_tm
    print("## QueryMatchAnyBuilder")
    print(f"search time: {search_tm}")
    print(condition)
    print(query.query_filter)
    show(response)
    print()

    condition = SearchCondition(color=["red blue"])
    query = QueryMatchAnyBuilder.build(condition=condition, vector=vector)
    start_tm = time.time()
    response = client.search(collection_name, query)
    search_tm = time.time() - start_tm
    print("## QueryMatchAnyBuilder")
    print(f"search time: {search_tm}")
    print(condition)
    print(query.query_filter)
    show(response)
    print()

    condition = SearchCondition(color=["red", "blue"])
    query = QueryBuilder.build(condition=condition, vector=vector)
    start_tm = time.time()
    response = client.search(collection_name, query)
    search_tm = time.time() - start_tm
    print("## QueryBuilder")
    print(f"search time: {search_tm}")
    print(condition)
    print(query.query_filter)
    show(response)
    print()

    condition = SearchCondition(color=["red", "blue"])
    query = QueryTextBuilder.build(condition=condition, vector=vector)
    start_tm = time.time()
    response = client.search(collection_name, query)
    search_tm = time.time() - start_tm
    print("## QueryTextBuilder")
    print(f"search time: {search_tm}")
    print(condition)
    print(query.query_filter)
    show(response)

    # delete index
    _ = client.delete_index(collection_name)
    print(f"index deleted: {collection_name}")
    print()

    print("DONE")


if __name__ == "__main__":
    main()
