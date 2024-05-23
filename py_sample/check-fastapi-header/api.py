from typing import List, Union

from fastapi import FastAPI, Header
from pydantic import BaseModel

app = FastAPI()


class Query(BaseModel):
    keyword: str


class SearchResult(BaseModel):
    items: List[str]


@app.get("/items")
def read_items(user_agent: Union[str, None]=Header(default=None)) -> dict:
    return {"User-Agent": user_agent}


@app.post("/search", response_model=SearchResult)
def search(query: Query, x_search_type: Union[str, None]=Header(default="base")) -> dict:
    items = [f"header={x_search_type}", f"query={query.keyword}"]
    return SearchResult(items=items)
