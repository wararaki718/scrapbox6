from pydantic import BaseModel, Field


class Question(BaseModel):
    text: str = Field(max_length=512)
