# generated by fastapi-codegen:
#   filename:  openapi.yaml
#   timestamp: 2023-07-30T11:48:31+00:00

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, Path

from .schema import Pets

app = FastAPI(
    version='1.0.0',
    title='Swagger Petstore',
    license={'name': 'MIT'},
    servers=[{'url': 'http://localhost:8000', 'description': 'local'}],
)


@app.get('/pets', response_model=Pets, tags=['pets'])
def list_pets(limit: Optional[int] = None) -> Pets:
    """
    List all pets
    """
    pass


@app.post('/pets', response_model=None, tags=['pets'])
def create_pets() -> None:
    """
    Create a pet
    """
    pass


@app.get('/pets/{pet_id}', response_model=Pets, tags=['pets'])
def show_pet_by_id(pet_id: str = Path(..., alias='petId')) -> Pets:
    """
    Info for a specific pet
    """
    pass
