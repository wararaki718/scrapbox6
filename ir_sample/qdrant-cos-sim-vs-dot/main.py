import gc
import time
from argparse import ArgumentParser, Namespace

from qdrant_client.models import (
    Distance,
    NamedVector,
    PointStruct,
    SearchParams,
    VectorParams,
)
from tqdm import tqdm

from client import SearchClient
from loader import NewsLoader
from normalizer import VectorNormalizer
from utils import show
from vectorizer import RandomVectorizer


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
        "cossim": VectorParams(
            size=256,
            distance=Distance.COSINE,
            on_disk=False,
        ),
        "dot": VectorParams(
            size=256,
            distance=Distance.DOT,
            on_disk=False,
        )
    }
    _ = client.create_index(
        collection_name,
        vectors_config=dense_vectors_config,
    )
    print(f"index created: {collection_name}")

    # modules
    random_vectorizer = RandomVectorizer(size=256)
    normalizer = VectorNormalizer()
    
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
        vector = random_vectorizer.generate()
        norm_vector = normalizer.normalize(vector)
        point = PointStruct(
            id=point_id,
            payload={"category": category},
            vector={"cossim": vector, "dot": norm_vector},
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

    n_try = 10
    for i in range(1, n_try+1):
        print(f"######### {i}-th #########")
        # cossim
        vector = random_vectorizer.generate()
        query_vector = NamedVector(
            name="cossim",
            vector=vector,
        )
        start_tm = time.time()
        results = client.search(collection_name, query_vector, search_params, limit=top_n)
        search_tm = time.time() - start_tm
        print(f"cos-sim: {search_tm}")
        # show(results)

        # dot product
        norm_vector = normalizer.normalize(vector)
        query_vector = NamedVector(
            name="dot",
            vector=norm_vector,
        )
        start_tm = time.time()
        results = client.search(collection_name, query_vector, search_params, limit=top_n)
        search_tm = time.time() - start_tm
        print(f"dot: {search_tm}")
        # show(results)

    # delete index
    _ = client.delete_index(collection_name)
    print(f"index deleted: {collection_name}")

    print("DONE")


if __name__ == "__main__":
    main()
