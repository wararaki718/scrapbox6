from builder import QueryBuilder
from client import SearchClient
from condition import SearchCondition
from utils import show


def main():
    collection_name = "sample"

    client = SearchClient()
    condition = SearchCondition(city="London")
    print(condition)
    print()

    query = QueryBuilder.build(condition=condition, vector=[0.9, 0.1, 0.1])
    response = client.search(collection_name, query)
    show(response)


if __name__ == "__main__":
    main()
