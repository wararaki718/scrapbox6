import os

import requests
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class Question(BaseModel):
    text: str


class Answer(BaseModel):
    text: str


HOST = os.getenv("BACKEND_HOST", default="localhost")
PORT = os.getenv("BACKEND_PORT", default="1234")


@app.post("/frontend", response_model=Answer)
def frontend(question: Question) -> Answer:
    url = f"http://{HOST}:{PORT}/backend"
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=question.dict()
    )
    result = response.json()

    text = f"frontend: {result['text']}"
    return Answer(text=text)
