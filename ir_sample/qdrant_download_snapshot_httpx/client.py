from pathlib import Path
from typing import List

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.models import Batch, ScoredPoint, UpdateResult, VectorParams, SnapshotDescription

from query import SearchQuery


class SearchClient:
    def __init__(self, host: str="localhost", port: int=6333):
        self._client = QdrantClient(host=host, port=port)

    def create_index(self, collection_name: str, params: VectorParams) -> bool:
        self._client.recreate_collection(
            collection_name=collection_name,
            vectors_config=params
        )
    
    def insert(self, collection_name: str, points: Batch) -> UpdateResult:
        response = self._client.upsert(
            collection_name=collection_name,
            points=points
        )
        return response

    def search(self, collection_name: str, query: SearchQuery) -> List[ScoredPoint]:
        response = self._client.search(
            collection_name=collection_name,
            **query.to_dict(),
            limit=10
        )
        return response

    def delete_index(self, collection_name: str) -> bool:
        response = self._client.delete_collection(collection_name=collection_name)
        return response

    def create_snapshot(self, collection_name: str) -> SnapshotDescription:
        response = self._client.create_snapshot(collection_name=collection_name)
        return response

    def download_snapshot(self, collection_name: str, snapshot_name: str, filepath: Path) -> bool:
        url = f"{self._client._client.rest_uri}/collections/{collection_name}/snapshots/{snapshot_name}"
        with httpx.stream("GET", url) as r:
            with open(filepath, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=8192):
                    f.write(chunk)
        return True
