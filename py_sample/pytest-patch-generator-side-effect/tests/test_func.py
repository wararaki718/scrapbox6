from typing import Generator
from unittest.mock import patch, MagicMock

import pytest

from app.func import CustomGenerator


@pytest.fixture
def generator() -> CustomGenerator:
    return CustomGenerator()


def generate_stub(self) -> MagicMock:
    stub = MagicMock(side_effect=[4, 5, 6])
    return stub


def test_generator(generator: CustomGenerator) -> None:
    for i, j in enumerate(generator.generate(), start=1):
        assert j == i


def test_patch_generator(generator: CustomGenerator) -> None:
    with patch("app.func.CustomGenerator.generate", generate_stub):
        for i, j in enumerate(generator.generate(), start=4):
            assert j == i
