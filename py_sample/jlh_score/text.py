from typing import List


class TextTokenizer:
    def tokenize(self, text: str) -> List[str]:
        return text.split()
