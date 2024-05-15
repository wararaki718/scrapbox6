import gc
from argparse import ArgumentParser, Namespace

from qdrant_client.models import (
    Distance,
    NamedVector,
    PointStruct,
    ScalarQuantizationConfig,
    ScalarType,
    SearchParams,
    VectorParams,
)
from tqdm import tqdm

from client import SearchClient
from loader import NewsLoader
from utils import show
from vectorizer import DenseVectorizer, RandomVectorizer


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--collection-name", default="newsgroups")
    return parser.parse_args()


def main():
    args = get_args()
    collection_name: str = args.collection_name
    print(f"collection name: {collection_name}")

    client = SearchClient()

    # create collection
    dense_vectors_config = {
        "dense-main": VectorParams(
            size=768,
            distance=Distance.COSINE,
            on_disk=False,
        ),
        "dense-sub": VectorParams(
            size=128,
            distance=Distance.COSINE,
            on_disk=False,
        )
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

    # dense_model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    # dense_vectorizer = DenseVectorizer(dense_model_name)
    random_vectorizer = RandomVectorizer()
    sub_random_vector = RandomVectorizer(size=128)
    
    # data load
    text_df = NewsLoader.load()
    print(f"the number of texts: {text_df.shape}")

    # insert
    points = []
    chunksize = 1000
    for point_id, (category, text) in tqdm(
        enumerate(zip(text_df.category, text_df.text)),
        total=len(text_df),
    ):
        # dense_vector = dense_vectorizer.transform(text)
        dense_vector = random_vectorizer.transform()
        sub_vector = sub_random_vector.transform()
        point = PointStruct(
            id=point_id,
            payload={"category": category},
            vector={"dense-main": dense_vector, "dense-sub": sub_vector},
        )
        points.append(point)
        if len(points) >= chunksize:
            client.upsert(collection_name, points)
            points = []
            gc.collect()

    if len(points) > 0:
        client.upsert(collection_name, points)
    print()

    # search
    print("search:")
    top_n = 10
    search_params = SearchParams(hnsw_ef=128, exact=False)

    # main
    dense_vector = random_vectorizer.transform()
    query_vector = NamedVector(
        name="dense-main",
        vector=dense_vector,
    )
    results = client.search(collection_name, query_vector, search_params, limit=top_n)
    print("dense result (main):")
    show(results)

    # sub
    dense_vector = sub_random_vector.transform()
    query_vector = NamedVector(
        name="dense-sub",
        vector=dense_vector,
    )
    results = client.search(collection_name, query_vector, search_params, limit=top_n)
    print("dense result (sub):")
    show(results)

    # delete index
    _ = client.delete_index(collection_name)
    print(f"index deleted: {collection_name}")

    print("DONE")


if __name__ == "__main__":
    main()
