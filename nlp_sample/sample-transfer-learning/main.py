import pandas as pd
from datasets import load_dataset

from model import NNModel
from vectorizer import TextVectorizer


def main() -> None:
    dataset_name = "dair-ai/emotion"
    dataset = load_dataset(dataset_name)
    print(f"train: {len(dataset['train'])}")
    print(f"valid: {len(dataset['validation'])}")
    print(f"test: {len(dataset['test'])}")

    print(type(dataset['train']))
    train_df = pd.DataFrame(dataset["train"])
    print(train_df.shape)

    model_name = "google-bert/bert-base-cased"
    vectorizer = TextVectorizer(model_name=model_name)

    X_train = []
    chunksize = 128
    for i in range(0, train_df.shape[0], chunksize):
        texts = train_df.iloc[i: i+chunksize].text
        #print(texts.shape)
        x = texts.apply(vectorizer.transform)
        break
    print(x.shape)
    print(len(x[0]))
    print(x[:3])

    n_input = 768
    n_hidden = 128
    n_output = 6
    model = NNModel(n_input, n_hidden, n_output)
    print("DONE")


if __name__ == "__main__":
    main()
