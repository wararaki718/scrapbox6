# generated by fastapi-codegen:
#   filename:  openapi.yaml
#   timestamp: 2023-07-30T11:48:31+00:00

from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, Path, status
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.responses import JSONResponse
from starlette.requests import Request

from .schema.model import Pet, Pets, UnprocessableEntityError, InternalServerError, NotFoundError, BadRequestError

app = FastAPI(
    version='1.0.0',
    title='Swagger Petstore',
    servers=[{'url': 'http://localhost:8000', 'description': 'local'}],
)

pets: List[Pets] = [
    Pet(id=10, name="test"),
    Pet(id=20, name="sample"),
    Pet(id=30, name="pet")
]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("request_validation_error")
    return JSONResponse(status_code=422, content={"msg": str(exc)})


@app.exception_handler(HTTPException)
async def validation_exception_handler(request: Request, exc: HTTPException):
    print("http error")
    return JSONResponse(status_code=exc.status_code, content={"msg": str(exc.detail)})


@app.get(
    '/pets',
    response_model=Pets,
    responses={
        "422": {"model": UnprocessableEntityError},
        "500": {"model": InternalServerError},
    }
)
def list_pets(limit: Optional[int] = None) -> Pets:
    """
    List all pets
    """
    if limit is not None:
        if limit < 0:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="out of range")
        return Pets(pets=pets)
    
    return Pets(pets=pets[:limit])


@app.post(
    '/pets',
    responses={
        "400": {"model": BadRequestError},
        "422": {"model": UnprocessableEntityError},
        "500": {"model": InternalServerError}
    }
)
def create_pets(pet: Pet) -> None:
    """
    Create a pet
    """
    pets.append(pet)
    return "created"


@app.get(
    '/pets/{pet_id}',
    response_model=Pet,
    responses={
        "404": {"model": NotFoundError},
        "422": {"model": UnprocessableEntityError},
        "500": {"model": InternalServerError}
    }
)
def show_pet_by_id(pet_id: str = Path(..., alias='pet_id')) -> Pet:
    """
    Info for a specific pet
    """
    try:
        target_id = int(pet_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="pet_id is not integer")
    
    for pet in pets:
        if pet.id == target_id:
            return pet

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"pet_id={target_id} is not found")