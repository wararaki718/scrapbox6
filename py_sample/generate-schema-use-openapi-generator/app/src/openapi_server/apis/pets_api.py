# coding: utf-8

from typing import Dict, List  # noqa: F401

from fastapi import (  # noqa: F401
    APIRouter,
    Body,
    Cookie,
    Depends,
    Form,
    Header,
    Path,
    Query,
    Response,
    Security,
    status,
)

from openapi_server.models.extra_models import TokenModel  # noqa: F401
from openapi_server.models.pet import Pet


router = APIRouter()


@router.post(
    "/pets",
    responses={
        201: {"description": "Null response"},
    },
    tags=["pets"],
    summary="Create a pet",
    response_model_by_alias=True,
)
async def create_pets(
) -> None:
    ...


@router.get(
    "/pets",
    responses={
        200: {"model": List[Pet], "description": "A paged array of pets"},
    },
    tags=["pets"],
    summary="List all pets",
    response_model_by_alias=True,
)
async def list_pets(
    limit: int = Query(None, description="How many items to return at one time (max 100)"),
) -> List[Pet]:
    ...


@router.get(
    "/pets/{petId}",
    responses={
        200: {"model": List[Pet], "description": "Expected response to a valid request"},
    },
    tags=["pets"],
    summary="Info for a specific pet",
    response_model_by_alias=True,
)
async def show_pet_by_id(
    petId: str = Path(None, description="The id of the pet to retrieve"),
) -> List[Pet]:
    ...
