# creating clear and convenient apis

from pathlib import Path


class CustomLogger:
    def __init__(self, filename: Path) -> None:
        self._filename = filename
    
    def __call__(self, msg: str) -> None:
        with open(self._filename, "a") as f:
            f.write(msg + "\n")
