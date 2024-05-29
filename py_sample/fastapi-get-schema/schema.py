from typing import List

from pydantic import BaseModel


class RequestBody(BaseModel):
    keyword: str


class ResponseBody(BaseModel):
    items: List[str]
