from typing import Callable


def sample_decorator(operator: Callable[..., int]) -> Callable[..., int]:
    def wrapper(x: int, y: int) -> int:
        print("start: sample")
        result = operator(x, y)
        print("end  : sample")
        return result
    return wrapper


def custom_decorator(operator: Callable[..., int]) -> Callable[..., int]:
    def wrapper(x: int, y: int) -> int:
        print("start: custom")
        result = operator(x, y)
        print("end  : custom")
        return result
    return wrapper
