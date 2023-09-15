from typing import Optional, Union

from qdrant_client.http.models import Filter, FieldCondition, MatchValue, IsEmptyCondition, PayloadField

from condition import SearchCondition


class CityFilterFactory:
    @classmethod
    def create(cls, condition: SearchCondition) -> Optional[Union[Filter, FieldCondition, IsEmptyCondition]]:
        if condition.city is not None and condition.is_empty_city:
            return Filter(
                should = [
                    FieldCondition(
                        key="city",
                        match=MatchValue(value=condition.city)
                    ),
                    IsEmptyCondition(
                        is_empty=PayloadField(key="city")
                    )
                ]
            )
        
        if condition.city is not None:
            return FieldCondition(
                key="city",
                match=MatchValue(value=condition.city)
            )

        if condition.is_empty_city:
            return IsEmptyCondition(
                is_empty=PayloadField(key="city")
            )
        
        return None
