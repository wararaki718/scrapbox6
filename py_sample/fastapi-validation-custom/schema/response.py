from typing import List

from pydantic import BaseModel


class Item(BaseModel):
    item_id: int
    name: str
    price: float


class SearchResult(BaseModel):
    items: List[Item]
