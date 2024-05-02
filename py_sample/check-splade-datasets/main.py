import json
import gc
from pathlib import Path

import pandas as pd


def main() -> None:
    msmarco_path = Path("/home/wararaki/workspace/splade/data/msmarco")
    collection = pd.read_csv(msmarco_path / "full_collection/raw.tsv", sep="\t", header=None)
    collection.columns = ["docid", "text"]
    print(collection.shape)
    print(collection.head(3))
    print()

    queries = pd.read_csv(msmarco_path / "train_queries/queries/raw.tsv", sep="\t", header=None)
    queries.columns = ["qid", "text"]
    print(queries.shape)
    print(queries.head(3))
    print()

    with open(msmarco_path / "train_queries/qrels.json") as f:
        qrels: dict = json.load(f)
    ids = set(queries.qid.astype(str)) & set(qrels.keys())
    print(len(qrels.keys()))
    print(len(ids))
    print()
    del queries
    gc.collect()

    qrels_ids = set()
    for value in qrels.values():
        qrels_ids |= set(value)
    ids = set(collection.docid.astype(str)) & qrels_ids
    print(len(qrels_ids))
    print(len(ids))
    del collection, qrels
    gc.collect()

    print("DONE")


if __name__ == "__main__":
    main()
