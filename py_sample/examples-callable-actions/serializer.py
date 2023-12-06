import json

import yaml


class JsonSerializer:
    def __call__(self, data: dict) -> str:
        return json.dumps(data)


class YamlSerializer:
    def __call__(self, data: dict) -> str:
        return yaml.safe_dump(data)


class DataSerializer:
    def __init__(self, serializer: callable) -> None:
        self._serializer = serializer
    
    def __call__(self, data: dict) -> str:
        return self._serializer(data)

    def switch_strategy(self, serializer: callable) -> None:
        self._serializer = serializer
