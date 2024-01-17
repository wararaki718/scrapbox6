import json
from pathlib import Path
from time import sleep

from opensearchpy import OpenSearch


def main() -> None:
    host = "localhost"
    port = 9200

    client = OpenSearch(hosts=[{"host": host, "port": port}])

    index_name = "my-knn-index-1"
    with open("./jsons/index.json") as f:
        index_body = json.load(f)
    response = client.indices.create(index_name, body=index_body)
    print("index created:")
    print(response)
    print()
    sleep(1)

    data = Path("./jsons/data.jsonl").read_text()
    response = client.bulk(data)
    print(f"data inserted: {response}")
    print()
    sleep(3)

    query = {
        "size": 2,
        "query": {
            "knn": {
                "my_vector2": {
                    "vector": [2, 3, 5, 6],
                    "k": 2,
                }
            }
        }
    }

    response = client.search(body=query, index=index_name)
    print(response)
    print()
    sleep(1)

    response = client.indices.delete(index_name)
    print(f"delete index: {response}")
    print()
    print("DONE")


if __name__ == "__main__":
    main()
