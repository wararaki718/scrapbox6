from typing import List

from pydantic import BaseModel


class Vector(BaseModel):
    values: List[float]
