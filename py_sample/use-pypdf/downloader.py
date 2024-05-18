from io import BytesIO

import httpx


class PDFDownloader:
    def download(self, uri: str, chunk_size: int=8192) -> BytesIO:
        f = BytesIO()
        with httpx.stream("GET", uri) as r:
            for chunk in r.iter_bytes(chunk_size=chunk_size):
                _ = f.write(chunk)
        f.seek(0)
        return f
