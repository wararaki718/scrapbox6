import httpx
from pathlib import Path


class Downloader:
    def download(self, url: str, filepath: Path) -> bool:
        with httpx.stream("GET", url) as response:
            with open(filepath, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
