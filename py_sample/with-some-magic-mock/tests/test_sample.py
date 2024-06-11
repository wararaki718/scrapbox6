from unittest.mock import MagicMock, patch

from app.sample import func_a, func_b, func_c


def func_stub() -> MagicMock:
    stub = MagicMock(return_value=1000)
    return stub


def test_func_a() -> None:
    assert func_a() == 1


def test_func_b() -> None:
    assert func_b() == 2


def test_func_c() -> None:
    assert func_c() == 3


def test_func_c_with_a_stub() -> None:
    with patch("app.sample.func_a", func_stub()):
        assert func_c() == 1002


def test_func_c_with_ab_stub() -> None:
    with patch("app.sample.func_a", func_stub()), patch("app.sample.func_b", func_stub()):
        assert func_c() == 2000
