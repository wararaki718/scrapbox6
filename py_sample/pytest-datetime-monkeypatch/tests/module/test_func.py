import datetime
from unittest.mock import MagicMock

from project.module.func import datediff


def datediff_stub() -> MagicMock:
    stub = MagicMock(wraps=datetime.datetime)
    stub.now.return_value = datetime.datetime(year=2020, month=2, day=2)
    return stub


def test_datediff(monkeypatch) -> None:
    monkeypatch.setattr(datetime, "datetime", datediff_stub())
    result = datediff(1)
    assert result == "2020-02-01"
