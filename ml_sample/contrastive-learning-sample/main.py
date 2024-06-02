from datasets import load_dataset
from sklearn.model_selection import train_test_split

from iterator import DataIterator
from train import Trainer
from utils import show_data
from vectorizer import DenseVectorizer


def main() -> None:
    # data load
    dataset_name = "bclavie/mmarco-japanese-hard-negatives"
    data = load_dataset(dataset_name)
    print(data.shape)
    print(data.column_names)
    print()

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
        data["train"]["positives"],
        data["train"]["negatives"],
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

    # vectorize
    ## chunking
    model_name = "tohoku-nlp/bert-base-japanese-v3"
    vectorizer = DenseVectorizer(model_name=model_name)
    # embeddings = vectorizer.transform(test_query[:3])
    #print(embeddings.shape)



    print("DONE")


if __name__ == "__main__":
    main()
