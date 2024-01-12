from fastapi import FastAPI

from schema.request import Text
from schema.response import Vector
from vectorizer import DenseVectorizer

app = FastAPI()


model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
vectorizer = DenseVectorizer(model_name)


@app.post("/vectorize", response_model=Vector)
def vectorize(request: Text) -> Vector:
    values = vectorizer.transform(request.content)
    return Vector(values=values)
