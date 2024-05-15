from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FilterSelector,
    Filter,
    NamedVector,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScoredPoint,
    SearchParams,
    UpdateResult,
    VectorParams,
)


class SearchClient:
    def __init__(self, host: str="localhost", port: int=6333):
        self._client = QdrantClient(host=host, port=port)

    def create_index(
            self,
            collection_name: str,
            vectors_config: Dict[str, VectorParams],
            quantization_config: ScalarQuantizationConfig,
        ) -> bool:
        self._client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            # quantization_config=ScalarQuantization(scalar=quantization_config),
            on_disk_payload=False,
        )
    
    def upsert(self, collection_name: str, points: List[PointStruct]) -> UpdateResult:
        response = self._client.upsert(
            collection_name=collection_name,
            points=points,
        )
        return response

    def search(self, collection_name: str, query_vector: NamedVector, search_params: SearchParams, limit: int=10) -> List[List[ScoredPoint]]:
        response = self._client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            search_params=search_params,
            limit=limit,
        )
        return response

    def delete_index(self, collection_name: str) -> bool:
        response = self._client.delete_collection(collection_name=collection_name)
        return response

    def delete_points(self, collection_name: str) -> UpdateResult:
        response = self._client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(filter=Filter())
        )
        return response
