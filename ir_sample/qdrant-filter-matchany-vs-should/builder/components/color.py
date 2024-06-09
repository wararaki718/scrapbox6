from typing import Optional, Union

from qdrant_client.http.models import Filter, FieldCondition, MatchAny, MatchValue

from condition import SearchCondition


class ColorFilterFactory:
    @classmethod
    def create(cls, condition: SearchCondition) -> Optional[Union[Filter, FieldCondition]]:
        if condition.color is None:
            return None

        if len(condition.color) == 0:
            return FieldCondition(
                key="color",
                match=MatchValue(value=condition.color[0])
            )

        return Filter(
            should = [
                FieldCondition(
                    key="color",
                    match=MatchValue(value=color)
                )
                for color in condition.color
            ]
        )


class ColorFilterMatchAnyFactory:
    @classmethod
    def create(cls, condition: SearchCondition) -> Optional[FieldCondition]:
        if condition.color is None:
            return None

        if len(condition.color) == 0:
            return FieldCondition(
                key="color",
                match=MatchValue(value=condition.color[0]),
            )

        return FieldCondition(
            key="color",
            match=MatchAny(any=condition.color),
        )
