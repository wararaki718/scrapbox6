from typing import List
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding


def main() -> None:
    documents: List[str] = [
        "This is built to be faster and lighter than other embedding libraries e.g. Transformers, Sentence-Transformers, etc.",
        "fastembed is supported by and maintained by Qdrant.",
    ]

    print("dense:")
    model_name = "BAAI/bge-small-en-v1.5"
    model: TextEmbedding = TextEmbedding(model_name=model_name)
    generator = model.embed(documents)
    for embedding in generator:
        print(embedding.shape)
    print()

    print("sparse:")
    model_name = "prithivida/Splade_PP_en_v1"
    model: SparseTextEmbedding = SparseTextEmbedding(model_name=model_name)
    generator = model.embed(documents)
    for embedding in generator:
        print(embedding.values.shape)
        print(embedding.indices.shape)
    print()

    print("late interaction:")
    model_name = "colbert-ir/colbertv2.0"
    model: LateInteractionTextEmbedding = LateInteractionTextEmbedding(model_name=model_name)
    generator = model.embed(documents)
    for embedding in generator:
        print(embedding.shape)
    print()
    print("DONE")


if __name__ == "__main__":
    main()
