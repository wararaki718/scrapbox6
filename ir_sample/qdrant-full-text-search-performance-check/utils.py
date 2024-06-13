from typing import Generator, List

from qdrant_client.http.models import ScoredPoint, PointStruct
from tqdm import tqdm

from generate import ConditionGenerator, VectorGenerator


def show(results: List[ScoredPoint]):
    print("[result]")
    for result in results:
        print(result)
    print("--------------------")
    print()


def get_data(
    n_data: int,
    n_color: int=64,
    n_vector: int=64,
    chunksize: int=1000,
) -> Generator[List[PointStruct], None, None]:
    color_generator = ConditionGenerator(n_color)
    vector_generator = VectorGenerator(n_vector)

    points = []
    for i in tqdm(range(1, n_data+1), total=n_data):
        payload = {
            "color": color_generator.generate(),
        }
        point = PointStruct(
            id=i,
            vector=vector_generator.generate(),
            payload=payload,
        )
        points.append(point)

        if len(points) == chunksize:
            yield points
            del points
            points = []

    if points:
        yield points


if __name__ == "__main__":
    data = list(get_data())
    print(data)
