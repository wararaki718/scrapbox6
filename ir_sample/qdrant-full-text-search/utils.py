from typing import List

from qdrant_client.http.models import ScoredPoint, PointStruct

from generate import VectorGenerator


def show(results: List[ScoredPoint]):
    print("[result]")
    for result in results:
        print(result)
    print("--------------------")
    print()


def get_data(
    n_vector: int=64,
) -> List[PointStruct]:
    vector_generator = VectorGenerator(n_vector)

    payloads = [
        {"color": "red green blue"},
        {"color": "red green blue orange"},
        {"color": "red green"},
        {"color": "red"},
    ]

    points = []
    for i, payload in enumerate(payloads, start=1):
        point = PointStruct(
            id=i,
            vector=vector_generator.generate(),
            payload=payload,
        )
        points.append(point)

    return points


if __name__ == "__main__":
    data = list(get_data())
    print(data)
