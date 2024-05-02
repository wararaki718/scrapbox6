from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from datasets import load_dataset


def main() -> None:
    dataset_name = "lifestyle"
    passages = load_dataset("colbertv2/lotte_passages", dataset_name)
    collection = [passage["text"] for passage in passages["dev_collection"]]

    data = load_dataset("colbertv2/lotte", dataset_name)
    queries = [query["query"] for query in data["search_dev"]]
    print(f"queries: {len(queries)}, passages: {len(collection)}")
    print()

    checkpoint_path = "colbert-ir/colbertv2.0"
    experiment_name = f"{dataset_name}.experiment"

    # index config
    nbits = 2
    max_id = 10000
    index_name = f"{dataset_name}.dev.{nbits}bits"
    config = ColBERTConfig(
        nbits=nbits,
        doc_maxlen=300,
        overwrite=True,
    )
    with Run().context(RunConfig(nranks=1, experiment=experiment_name)):
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name=index_name, collection=collection[:max_id], overwrite=True)
    print(f"indexed: {indexer.get_index()}")
    print()

    answer_pids = [query["answer"]["answer_pids"] for query in data["search_dev"]]
    filtered_queries = [
        query for query, pids in zip(queries, answer_pids)
        if any(pid < max_id for pid in pids)
    ]
    with Run().context(RunConfig(experiment=experiment_name)):
        searcher = Searcher(index=index_name, collection=collection)

    query = filtered_queries[13]
    print(f"query > {query}")

    results = searcher.search(query, k=3)
    for pid, rank, score in zip(*results):
        passage = searcher.collection[pid]
        print(f"{rank}: {score} > {passage}")
    print()

    print("DONE")


if __name__=='__main__':
    main()
