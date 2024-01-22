from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class Question(BaseModel):
    text: str


class Answer(BaseModel):
    text: str


@app.post("/backend", response_model=Answer)
def backend(question: Question) -> Answer:
    text = f"backend: {question.text}"
    return Answer(text=text)
