import time

from qdrant_client.models import (
    Distance,
    NamedVector,
    PointStruct,
    ScalarQuantizationConfig,
    ScalarType,
    SearchParams,
    VectorParams,
)

from client import SearchClient
from utils import get_texts, show
from vectorizer import DenseVectorizer


def main():
    collection_name = "sample"
    client = SearchClient()

    # create collection
    dense_vectors_config = {
        "dense": VectorParams(
            size=768,
            distance=Distance.COSINE,
            on_disk=False,
        ),
    }
    quantization_config = ScalarQuantizationConfig(
        type=ScalarType.INT8,
        quantile=0.99,
        always_ram=True,
    )
    _ = client.create_index(
        collection_name,
        vectors_config=dense_vectors_config,
        quantization_config=quantization_config
    )
    print(f"index created: {collection_name}")

    dense_model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    dense_vectorizer = DenseVectorizer(dense_model_name)
    
    # data load
    texts = get_texts()

    # insert
    points = []
    for point_id, text in enumerate(texts):
        dense_vector = dense_vectorizer.transform(text)
        point = PointStruct(id=point_id, payload={}, vector={"dense": dense_vector})
        points.append(point)

    client.upsert(collection_name, points)
    print(f"data inserted: {len(points)}")
    print()

    # search
    print("search:")
    top_n = 10
    dense_vector = dense_vectorizer.transform(text[0])
    query_vector = NamedVector(
        name="dense",
        vector=dense_vector,
    )
    search_params = SearchParams(hnsw_ef=128, exact=False)
    results = client.search(collection_name, query_vector, search_params, limit=top_n)

    print("dense result:")
    show(results)

    time.sleep(3)    

    # delete index
    _ = client.delete_index(collection_name)
    print(f"index deleted: {collection_name}")

    print("DONE")


if __name__ == "__main__":
    main()
