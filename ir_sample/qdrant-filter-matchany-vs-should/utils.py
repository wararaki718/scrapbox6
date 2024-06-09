from typing import Generator, List

from qdrant_client.http.models import ScoredPoint, PointStruct
from tqdm import tqdm

from generate import ConditionGenerator, VectorGenerator


def show(results: List[ScoredPoint]):
    for result in results:
        print(result)
    print()


def get_data(
    n: int=100,
    n_color: int=32,
    n_city: int=64,
    n_vector: int=64,
    chunksize: int=10,
) -> Generator[List[PointStruct], None, None]:
    color_generator = ConditionGenerator(n_color)
    city_generator = ConditionGenerator(n_city)
    vector_generator = VectorGenerator(n_vector)

    points = []
    for i in tqdm(range(1, n+1), total=n):
        payload = {
            "color": color_generator.generate(),
            "city" : city_generator.generate(),
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
    
    if len(points) > 0:
        yield points


if __name__ == "__main__":
    data = list(get_data(n=10, chunksize=1))
    print(data)
