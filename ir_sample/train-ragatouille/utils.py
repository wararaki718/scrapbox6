import random


def make_pairs(queries: list, documents: list) -> list:
    pairs = []
    for query in queries:
        docs = random.sample(documents, 10)
        for doc in docs:
            pairs.append((query, doc))
    return pairs
