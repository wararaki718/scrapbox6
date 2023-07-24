from typing import List

from qdrant_client.http.models import Filter, SearchParams

from .components import CityFilterFactory, ColorFilterFactory
from condition import SearchCondition
from query import SearchQuery


class QueryBuilder:
    @classmethod
    def build(cls, condition: SearchCondition, vector: List[float], ef: int=128) -> SearchQuery:
        filters = [
            CityFilterFactory.create(condition),
            ColorFilterFactory.create(condition)
        ]

        must = list(filter(lambda x: x is not None, filters))

        if len(must) > 0:
            query_filter = Filter(
                must=must
            )
        else:
            query_filter = None

        search_params = SearchParams(
            hnsw_ef=ef,
            exact=False
        )

        return SearchQuery(
            query_filter=query_filter,
            search_params=search_params,
            query_vector=vector
        )
