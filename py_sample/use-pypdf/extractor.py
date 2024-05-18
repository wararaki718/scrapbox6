from io import BytesIO
from typing import List

import pypdf


class PDFTextExtractor:
    def extract(self, pdf: BytesIO) -> List[str]:
        reader = pypdf.PdfReader(pdf)
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            texts.append(text)
        return texts
