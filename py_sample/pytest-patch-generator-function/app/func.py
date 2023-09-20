from typing import Generator


class CustomGenerator:
    def generate(self) -> Generator[int, None, None]:
        yield 1
        yield 2
        yield 3
