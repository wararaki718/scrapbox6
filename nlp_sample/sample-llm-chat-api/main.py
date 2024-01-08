from fastapi import FastAPI

from schema.request import Question
from schema.response import Answer
from service import LLMService


app = FastAPI()

service = LLMService()

@app.get("/ping")
def ping() -> str:
    return "pong"


@app.post("/chat", response_model=Answer)
def chat(question: Question) -> Answer:
    return service.generate(question)
