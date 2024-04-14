from ragatouille import RAGTrainer
from ragatouille.data import CorpusProcessor, llama_index_sentence_splitter

from loader import WikipediaPageLoader, QueryLoader
from utils import make_pairs

def main() -> None:
    # model definition
    model_name = "GhibliColBERT"
    pretrained_model_name="colbert-ir/colbertv2.0"
    language_code="en"
    trainer = RAGTrainer(
        model_name=model_name,
        pretrained_model_name=pretrained_model_name,
        language_code=language_code,
    )
    print("model defined.")

    url = "https://en.wikipedia.org/w/api.php"
    corpus = [
        WikipediaPageLoader.load(url, title="Hayao_Miyazaki"),
        WikipediaPageLoader.load(url, title="Studio_Ghibli"),
        WikipediaPageLoader.load(url, title="Toei_Animation"),
    ]
    print(len(corpus[0]))
    print(len(corpus[1]))
    print(len(corpus[2]))
    print()

    chunksize = 256
    processor = CorpusProcessor(
        document_splitter_fn=llama_index_sentence_splitter,
    )
    documents = processor.process_corpus(corpus, chunk_size=chunksize)
    print(len(documents))
    print()

    queries = QueryLoader.load()
    print(len(queries))
    print()

    output_path = "./data/"
    pairs = make_pairs(queries, documents)
    trainer.prepare_training_data(
        raw_data=pairs,
        data_out_path=output_path,
        all_documents=corpus,
        num_new_negatives=10,
        mine_hard_negatives=True,
    )
    print(f"save train data: {output_path}")
    print()

    trainer.train(
        batch_size=32,
        nbits=4,
        # maxsteps=500000,
        maxsteps=100,
        use_ib_negatives=True,
        dim=128,
        learning_rate=5e-6,
        doc_maxlen=256,
        use_relu=False,
        warmup_steps="auto",
    )
    print("train finished.")
    print()

    print("DONE")


if __name__ == "__main__":
    main()
