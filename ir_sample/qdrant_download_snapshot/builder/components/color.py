from typing import Optional, Union

from qdrant_client.http.models import Filter, FieldCondition, MatchValue, IsEmptyCondition, PayloadField

from condition import SearchCondition


class ColorFilterFactory:
    @classmethod
    def create(cls, condition: SearchCondition) -> Optional[Union[Filter, FieldCondition, IsEmptyCondition]]:
        if condition.color is not None and condition.is_empty_color:
            return Filter(
                should = [
                    FieldCondition(
                        key="color",
                        match=MatchValue(value=condition.color)
                    ),
                    IsEmptyCondition(
                        is_empty=PayloadField(key="color")
                    )
                ]
            )
        
        if condition.color is not None:
            return FieldCondition(
                key="color",
                match=MatchValue(value=condition.color)
            )

        if condition.is_empty_color:
            return IsEmptyCondition(
                is_empty=PayloadField(key="color")
            )
        
        return None
