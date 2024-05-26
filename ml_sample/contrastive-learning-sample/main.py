from datasets import load_dataset
from sklearn.model_selection import train_test_split

from vectorizer import DenseVectorizer


def main() -> None:
    # use
    dataset_name = "bclavie/mmarco-japanese-hard-negatives"
    data = load_dataset(dataset_name)
    print(data.shape)
    print(data.column_names)
    print()

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
    print(len(train_query))
    print(len(train_positive_documents))
    print(len(train_negative_documents))
    print()

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
    print(len(valid_query))
    print(len(valid_positive_documents))
    print(len(valid_negative_documents))
    print()

    print("test dataset:")
    print(len(test_query))
    print(len(test_positive_documents))
    print(len(test_negative_documents))
    print()

    model_name = "tohoku-nlp/bert-base-japanese-v3"
    vectorizer = DenseVectorizer(model_name=model_name)
    embeddings = vectorizer.transform(test_query[0])
    print(embeddings.shape)

    embeddings = vectorizer.transform_batch(test_query[:3])
    print(embeddings.shape)

    print("DONE")


if __name__ == "__main__":
    main()
