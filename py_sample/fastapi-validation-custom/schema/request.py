from pydantic import BaseModel, validator


class Query(BaseModel):
    keyword: str

    @validator("keyword")
    def keyword_minimum_length(cls, v: str) -> str:
        if len(v) < 3:
            raise ValueError("keyword must be 3 or more characters")
        return v
    
    @validator("keyword")
    def keyword_maximum_length(cls, v: str) -> str:
        if len(v) > 20:
            raise ValueError("keyword must be 20 or less characters")
        return v
