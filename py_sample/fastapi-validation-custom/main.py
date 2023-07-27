from fastapi import FastAPI

from schema.request import Query
from schema.response import SearchResult
from service import SearchService


app = FastAPI()


@app.get("/ping")
def ping() -> str:
    return "pong"


@app.post("/search", response_model=SearchResult)
def search(query: Query) -> SearchResult:
    items = SearchService.search(query)
    return SearchResult(items=items)
