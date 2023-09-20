from typing import Generator
from unittest.mock import patch

import pytest

from app.func import CustomGenerator


@pytest.fixture
def generator() -> CustomGenerator:
    return CustomGenerator()


def generate_stub(self) -> Generator[int, None, None]:
    yield 4
    yield 5
    yield 6


def test_generator(generator: CustomGenerator) -> None:
    for i, j in enumerate(generator.generate(), start=1):
        assert j == i


def test_patch_generator(generator: CustomGenerator) -> None:
    with patch("app.func.CustomGenerator.generate", generate_stub):
        for i, j in enumerate(generator.generate(), start=4):
            assert j == i
