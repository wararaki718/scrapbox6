from fastapi import FastAPI

from schema import RequestBody, ResponseBody


app = FastAPI()


@app.get("/ping")
def ping() -> str:
    return "pong"


@app.post("/sample", response_model=ResponseBody)
def sample(request: RequestBody) -> ResponseBody:
    items = list(request.keyword)
    return ResponseBody(items=items)
