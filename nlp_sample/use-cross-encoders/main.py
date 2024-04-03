import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder


def main() -> None:
    model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

    query = "A man is eating pasta."
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

    ranks = model.rank(query, corpus)

    print(f"Query: {query}")
    for rank in ranks:
        print(f"{rank['score']:.2f}\t{corpus[rank['corpus_id']]}")
    print()
    
    sentence_combinations = [[query, sentence] for sentence in corpus]
    scores = model.predict(sentence_combinations)

    ranked_indices = np.argsort(scores)[::-1]
    print(f"scores : {scores}")
    print(f"indices: {ranked_indices}")

    print("DONE")


if __name__ == "__main__":
    main()
