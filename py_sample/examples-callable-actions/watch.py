# timing.py

import time
from typing import Any


class ExecutionTimer:
    def __init__(self, func: callable) -> None:
        self._func = func
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        start = time.perf_counter()
        result = self._func(*args, **kwds)
        end = time.perf_counter()
        print(f"{self._func.__name__}: {(end - start) * 1000} ms")
        return result
