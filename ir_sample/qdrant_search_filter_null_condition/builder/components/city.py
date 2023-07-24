from typing import Optional, Union

from qdrant_client.http.models import Filter, FieldCondition, MatchValue, IsNullCondition, PayloadField

from condition import SearchCondition


class CityFilterFactory:
    @classmethod
    def create(cls, condition: SearchCondition) -> Optional[Union[Filter, FieldCondition, IsNullCondition]]:
        if condition.city is not None and condition.is_null_city:
            return Filter(
                should = [
                    FieldCondition(
                        key="city",
                        match=MatchValue(value=condition.city)
                    ),
                    IsNullCondition(
                        is_null=PayloadField(key="city")
                    )
                ]
            )
        
        if condition.city is not None:
            return FieldCondition(
                key="city",
                match=MatchValue(value=condition.city)
            )

        if condition.is_null_city:
            return IsNullCondition(
                is_null=PayloadField(key="city")
            )
        
        return None
