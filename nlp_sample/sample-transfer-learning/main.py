import gc

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from model import NNModel
from vectorizer import TextVectorizer
from utils import try_gpu


def main() -> None:
    dataset_name = "dair-ai/emotion"
    dataset = load_dataset(dataset_name)
    print(f"train: {len(dataset['train'])}")
    print(f"valid: {len(dataset['validation'])}")
    print(f"test: {len(dataset['test'])}")

    train_df = pd.DataFrame(dataset["train"])
    print(train_df.shape)

    model_name = "google-bert/bert-base-cased"
    vectorizer = TextVectorizer(model_name=model_name)
    X_train = []
    chunksize = 64
    for i in tqdm(range(0, train_df.shape[0], chunksize)):
        texts = train_df.iloc[i: i+chunksize].text.tolist()
        x = vectorizer.transform(texts)
        X_train.append(x)
    print(len(X_train))
    del vectorizer
    gc.collect()

    n_input = 768
    n_hidden = 128
    n_output = 6
    model = NNModel(n_input, n_hidden, n_output)
    model = try_gpu(model)
    
    print("DONE")


if __name__ == "__main__":
    main()
