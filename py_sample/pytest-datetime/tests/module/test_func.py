from datetime import datetime
from unittest.mock import MagicMock, patch

from project.module.func import datediff


def datediff_stub() -> MagicMock:
    stub = MagicMock()
    stub.now.return_value = datetime(year=2020, month=2, day=2)
    return stub


def test_datediff() -> None:
    with patch("project.module.func.datetime", datediff_stub()):
        result = datediff(1)
    assert result == "2020-02-01"
