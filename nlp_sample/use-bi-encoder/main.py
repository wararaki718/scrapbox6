import torch
from sentence_transformers import SentenceTransformer, util


def main() -> None:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin.",
        "Two men pushed carts through the woods.",
        "A man is riding a white horse on an enclosed ground.",
        "A monkey is playing drums.",
        "A cheetah is running behind its prey.",
    ]
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    print(f"the number of corpus embeddings: {corpus_embeddings.shape}")

    queries = [
        "A man is eating pasta.",
        "Someone in a gorilla costume is playing a set of drums.",
        "A cheetah chases prey on across a field.",
    ]
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    print(f"the number of queries embeddings: {query_embeddings.shape}")
    print()

    top_k: int = min(5, len(corpus))
    for query, query_embedding in zip(queries, query_embeddings):
        similarities = util.cos_sim(query_embedding, corpus_embeddings)[0]
        (scores, indices) = torch.topk(similarities, k=top_k)

        print(f"query: {query}")
        for i, (score, index) in enumerate(zip(scores, indices), start=1):
            print(f"{i}-th: {corpus[index]}, (score={score:.4f})")
        print()
    print("DONE")


if __name__ == "__main__":
    main()
