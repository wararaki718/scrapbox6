from typing import Optional, Union

from qdrant_client.http.models import Filter, FieldCondition, MatchValue, IsNullCondition, PayloadField

from condition import SearchCondition


class ColorFilterFactory:
    @classmethod
    def create(cls, condition: SearchCondition) -> Optional[Union[Filter, FieldCondition, IsNullCondition]]:
        if condition.color is not None and condition.is_null_color:
            return Filter(
                should = [
                    FieldCondition(
                        key="color",
                        match=MatchValue(value=condition.color)
                    ),
                    IsNullCondition(
                        is_null=PayloadField(key="color")
                    )
                ]
            )
        
        if condition.color is not None:
            return FieldCondition(
                key="color",
                match=MatchValue(value=condition.color)
            )

        if condition.is_null_color:
            return IsNullCondition(
                is_null=PayloadField(key="color")
            )
        
        return None
