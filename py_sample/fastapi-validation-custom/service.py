from typing import List

from repository import ItemRepository
from schema.request import Query
from schema.response import Item


class SearchService:
    @classmethod
    def search(cls, query: Query) -> List[Item]:
        keyword = query.keyword
        items = ItemRepository.search(keyword)
        return items
