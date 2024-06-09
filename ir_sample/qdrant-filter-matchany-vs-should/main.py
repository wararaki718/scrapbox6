import time
from qdrant_client.models import VectorParams

from builder import QueryBuilder, QueryMatchAnyBuilder
from client import SearchClient
from condition import SearchCondition
from generate import ConditionGenerator, VectorGenerator
from utils import get_data, show


def main():
    collection_name = "sample"
    dim = 128
    n_color = 32
    n_city = 64

    client = SearchClient()
    params = VectorParams(
        size=dim,
        distance="Cosine",
    )
    _ = client.create_index(collection_name, params)
    print(f"index created: {collection_name}")

    # data
    n_data = 100000
    chunksize = 1000
    for points in get_data(n_data, n_color, n_city, dim, chunksize):
        client.insert(collection_name, points)
    print(f"data inserted: {n_data}")

    # generators
    color_generator = ConditionGenerator(n_color)
    city_generator = ConditionGenerator(n_city)
    vector_generator = VectorGenerator(dim)

    # search
    n_try = 10
    for i in range(1, n_try+1):
        print(f"{i}-th:")
        condition = SearchCondition(
            city=city_generator.generate(),
            color=color_generator.generate(),
        )
        vector = vector_generator.generate()
        print(condition)

        # should
        query = QueryBuilder.build(condition=condition, vector=vector)
        start_tm = time.time()
        response = client.search(collection_name, query)
        search_tm = time.time() - start_tm
        print(f"'should condition' search time: {search_tm}")
        # show(response)

        # matchany
        query = QueryMatchAnyBuilder.build(condition=condition, vector=vector)
        start_tm = time.time()
        response = client.search(collection_name, query)
        search_tm = time.time() - start_tm
        print(f"'matchany condition' search time: {search_tm}")
        # show(response)
        print()
    
    _ = client.delete_index(collection_name)
    print(f"index deleted: {collection_name}")

    print("DONE")


if __name__ == "__main__":
    main()
