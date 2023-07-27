from typing import List

from schema.response import Item


class Database:
    def __init__(self) -> None:
        items = [
            {"item_id": 1, "name": "Foo", "price": 50.2},
            {"item_id": 2, "name": "Bar", "price": 62},
            {"item_id": 3, "name": "Baz", "price": 50.2},
            {"item_id": 4, "name": "Foo", "price": 62},
            {"item_id": 5, "name": "Foo", "price": 50.2},
            {"item_id": 6, "name": "Bar", "price": 62},
            {"item_id": 7, "name": "Baz", "price": 50.2},
            {"item_id": 8, "name": "Foo", "price": 62},
            {"item_id": 9, "name": "Foo", "price": 50.2},
            {"item_id": 10, "name": "Bar", "price": 62},
        ]

        self._db = [Item(**item) for item in items]

    def search(self, keyword: str) -> List[Item]:
        return [item for item in self._db if item.name == keyword]
