from ragatouille import RAGPretrainedModel

from loader import WikipediaPageLoader


def show(results: list[dict]) -> None:
    for i, result in enumerate(results, start=1):
        print(f"{i}-th doucment:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    print()


def main() -> None:
    model_name = "colbert-ir/colbertv2.0"
    rag = RAGPretrainedModel.from_pretrained(model_name)

    url = "https://en.wikipedia.org/w/api.php"
    title = "Hayao_Miyazaki"
    documents: list = WikipediaPageLoader.load(url, title)
    print(f"n_documents: {len(documents)}")

    index_name = "Miyazaki"
    document_id = "miyazaki"
    rag.index(
        collection=[documents],
        document_ids=[document_id],
        document_metadatas=[{"entity": "person", "source": "wikipedia"}],
        index_name=index_name,
        max_document_length=180,
        split_documents=True,
    )
    print("indexed dataset.")
    print()

    k = 3
    query = "What animation studio did Miyazaki found?"
    results = rag.search(query=query, k=k)
    print(f"query: {query}")
    show(results)

    query = "Miyazaki son name"
    results = rag.search(query=query, k=k)
    print(f"query: {query}")
    show(results)

    print("DONE")


if __name__ == "__main__":
    main()
