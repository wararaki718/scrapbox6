from dataclasses import dataclass
from typing import List


@dataclass
class SearchCondition:
    color: List[str]
