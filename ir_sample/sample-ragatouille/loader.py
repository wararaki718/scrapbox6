import requests


class WikipediaPageLoader:
    @classmethod
    def load(cls, url: str, title: str) -> list:
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        }

        headers = {
            "User-Agent": "RAGautoille_tutorial/0.0.1",
        }

        response = requests.get(url, params=params, headers=headers)
        data = response.json()

        page = next(iter(data["query"]["pages"].values()))
        return page["extract"] if "extract" in page else None
