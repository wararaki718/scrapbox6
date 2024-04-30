from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer


def main() -> None:
    config = ColBERTConfig(
        nbits=2,
        root="experiments",
        overwrite=True,
    )

    checkpoint_path = "checkpoint"
    collection_path = "data/collections.tsv"
    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name="msmarco.nbits=2", collection=collection_path, overwrite=True)

    print("DONE")


if __name__=='__main__':
    main()
