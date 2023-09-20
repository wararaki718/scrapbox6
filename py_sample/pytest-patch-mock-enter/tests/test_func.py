from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app.func import Downloader


@pytest.fixture
def downloader() -> Downloader:
    return Downloader()


def stream_stub(method: str, url: str) -> MagicMock:
    response = MagicMock()
    response.iter_bytes = MagicMock(return_value=[b"hello"])

    stub = MagicMock()
    stub.__enter__ = MagicMock(return_value=response)
    
    return stub


def test_downloader(downloader: Downloader, tmp_path: Path) -> None:
    url = "https://example.com/"
    filepath = tmp_path / "downloader.html"
    downloader.download(url, filepath)

    assert filepath.read_text() == Path("resources/sample.html").read_text()


def test_patch_downloader(downloader: Downloader, tmp_path: Path) -> None:
    url = "dummy"
    filepath = tmp_path / "sample.html"
    with patch("httpx.stream", stream_stub):
        downloader.download(url, filepath)

    assert filepath.read_text() == "hello"
