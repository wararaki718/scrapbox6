from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchCondition:
    color: Optional[str] = None
    city: Optional[str] = None
    is_null_color: bool = False
    is_null_city: bool = False
