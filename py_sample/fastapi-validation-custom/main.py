from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError, ValidationError
from fastapi.responses import JSONResponse

from schema.request import Query
from schema.response import SearchResult, IndexResult
from service import SearchService, IndexService


app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> str:
    return JSONResponse(
       status_code=status.HTTP_400_BAD_REQUEST,
       content=jsonable_encoder({"detail": exc.errors()})
    )


@app.get("/ping")
def ping() -> str:
    return "pong"


@app.post("/search", response_model=SearchResult)
def search(query: Query) -> SearchResult:
    items = SearchService.search(query)
    return SearchResult(items=items)


@app.get("/index", response_model=IndexResult)
def index() -> IndexResult:
    indices = IndexService.index()
    return IndexResult(indices=indices)
