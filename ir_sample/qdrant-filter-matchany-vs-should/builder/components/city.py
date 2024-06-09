from typing import Optional, Union

from qdrant_client.http.models import Filter, FieldCondition, MatchAny, MatchValue

from condition import SearchCondition


class CityFilterFactory:
    @classmethod
    def create(cls, condition: SearchCondition) -> Optional[Union[Filter, FieldCondition]]:
        if condition.city is None:
            return None
        
        if len(condition.city) == 0:
            return FieldCondition(
                key="city",
                match=MatchValue(value=condition.city[0]),
            )

        return Filter(
            should = [
                FieldCondition(
                    key="city",
                    match=MatchValue(value=city),
                )
                for city in condition.city
            ]
        )


class CityFilterMatchAnyFactory:
    @classmethod
    def create(cls, condition: SearchCondition) -> Optional[FieldCondition]:
        if condition.city is None:
            return None
        
        if len(condition.city) == 0:
            return FieldCondition(
                key="city",
                match=MatchValue(value=condition.city[0]),
            )

        return FieldCondition(
            key="city",
            match=MatchAny(any=condition.city),
        )
