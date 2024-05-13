from typing import List

from qdrant_client.models import ScoredPoint


def show(points: List[ScoredPoint]) -> None:
    for rank, point in enumerate(points, start=1):
        print(f"{rank}-th: id={point.id}, score={point.score}")
    print()
