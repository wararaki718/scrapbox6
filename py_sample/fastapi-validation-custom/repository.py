from typing import List

from db import Database
from schema.response import Item


database = Database()


class ItemRepository:
    @classmethod
    def search(cls, key: str) -> List[Item]:
        items = database.search(key)
        return items

    @classmethod
    def index(cls) -> List[str]:
        indices = database.index()
        return indices
