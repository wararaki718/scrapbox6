# generated by datamodel-codegen:
#   filename:  openapi.yaml
#   timestamp: 2023-07-31T13:42:18+00:00

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel


class Pet(BaseModel):
    id: int
    name: str
    tag: Optional[str] = None


class Pets(BaseModel):
    __root__: List[Any]


class Error(BaseModel):
    code: int
    message: str
