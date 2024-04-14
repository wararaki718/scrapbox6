import random
from typing import Optional

import requests


class WikipediaPageLoader:
    @classmethod
    def load(cls, url: str, title: str) -> Optional[list]:
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        }

        headers = {
            "User-Agent": "RAGatouille_tutorial/0.0.1"
        }

        response = requests.get(url, params=params, headers=headers)
        data = response.json()

        page = next(iter(data["query"]["pages"].values()))
        return page["extract"] if "extract" in page else None


class QueryLoader:
    @classmethod
    def load(cls) -> list:
        queries = [
            "What manga did Hayao Miyazaki write?",
            "which film made ghibli famous internationally",
            "who directed Spirited Away?",
            "when was Hikotei Jidai published?",
            "where's studio ghibli based?",
            "where is the ghibli museum?",
        ] * 3
        return queries
