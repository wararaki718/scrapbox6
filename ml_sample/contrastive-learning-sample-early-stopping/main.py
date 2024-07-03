import gc
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from evaluate import TripletEvaluator
from model import TripletModel
from train import Trainer
from utils import show_data, try_gpu
from vectorizer import DenseVectorizer


def main() -> None:
    # data load
    dataset_name = "bclavie/mmarco-japanese-hard-negatives"
    data = load_dataset(dataset_name)
    print(data.shape)
    print(data.column_names)
    print()

    positive_documents = [document[0] for document in data["train"]["positives"]]
    negative_documents = [document[0] for document in data["train"]["negatives"]]

    # train test split
    (
        train_query,
        test_query,
        train_positive_documents,
        test_positive_documents,
        train_negative_documents,
        test_negative_documents
    ) = train_test_split(
        data["train"]["query"],
        positive_documents,
        negative_documents,
        test_size=0.2,
        random_state=42,
    )
    print("train datasets:")
    show_data(train_query, train_positive_documents, train_negative_documents)

    # valid test split
    (
        valid_query,
        test_query,
        valid_positive_documents,
        test_positive_documents,
        valid_negative_documents,
        test_negative_documents
    ) = train_test_split(
        test_query,
        test_positive_documents,
        test_negative_documents,
        test_size=0.5,
        random_state=42,
    )
    print("validation dataset:")
    show_data(valid_query, valid_positive_documents, valid_negative_documents)

    print("test dataset:")
    show_data(test_query, test_positive_documents, test_negative_documents)
    del data, positive_documents, negative_documents
    gc.collect()

    # vectorize
    ## chunking
    model_name = "tohoku-nlp/bert-base-japanese-v3"
    vectorizer = DenseVectorizer(model_name=model_name)
    X_train_queries = vectorizer.transform(train_query[:256], chunksize=64)
    X_train_positive_documents = vectorizer.transform(train_positive_documents[:256], chunksize=64)
    X_train_negative_documents = vectorizer.transform(train_negative_documents[:256], chunksize=64)
    print(len(X_train_queries))
    print(X_train_queries[0].shape)
    print()

    X_valid_queries = vectorizer.transform(valid_query[:256], chunksize=64)
    X_valid_positive_documents = vectorizer.transform(valid_positive_documents[:256], chunksize=64)
    X_valid_negative_documents = vectorizer.transform(valid_negative_documents[:256], chunksize=64)
    print(len(X_valid_queries))
    print(X_valid_queries[0].shape)
    print()

    # train
    model = TripletModel(
        n_query_input=X_train_queries[0].shape[1],
        n_document_input=X_train_positive_documents[0].shape[1],
        n_query_output=128,
        n_document_output=128,
    )
    model = try_gpu(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    trainer = Trainer()
    trainer.train(
        model,
        optimizer,
        X_train_queries,
        X_train_positive_documents,
        X_train_negative_documents,
        X_valid_queries,
        X_valid_positive_documents,
        X_valid_negative_documents,
    )
    del X_train_queries, X_train_positive_documents, X_train_negative_documents
    del X_valid_queries, X_valid_positive_documents, X_valid_negative_documents
    gc.collect()
    print("model trained.")

    # evaluate
    X_test_queries = vectorizer.transform(test_query[:256], chunksize=64)
    X_test_positive_documents = vectorizer.transform(test_positive_documents[:256], chunksize=64)
    X_test_negative_documents = vectorizer.transform(test_negative_documents[:256], chunksize=64)
    print(len(X_test_queries))
    print(X_test_queries[0].shape)
    print()

    evaluator = TripletEvaluator()
    result = evaluator.evaluate(model, X_test_queries, X_test_positive_documents, X_test_negative_documents)
    print(f"accuracy: {result}")
    print()

    print("DONE")


if __name__ == "__main__":
    main()
