import gc

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

from executer import TrainValidExecuter
from model import NNModel
from utils import try_gpu
from vectorizer import TextVectorizer, LabelVectorizer


def main() -> None:
    dataset_name = "dair-ai/emotion"
    dataset = load_dataset(dataset_name)
    print(f"train: {len(dataset['train'])}")
    print(f"valid: {len(dataset['validation'])}")
    print(f"test: {len(dataset['test'])}")

    model_name = "google-bert/bert-base-cased"
    chunksize = 32
    text_vectorizer = TextVectorizer(model_name=model_name)
    label_vectorizer = LabelVectorizer()

    train_df = pd.DataFrame(dataset["train"])
    X_train = text_vectorizer.transform(train_df.text, chunksize)
    y_train = label_vectorizer.transform(train_df.label)
    print(f"train: {len(X_train)}")
    del train_df
    gc.collect()

    valid_df = pd.DataFrame(dataset["validation"])
    X_valid = text_vectorizer.transform(train_df.text, chunksize)
    y_valid = label_vectorizer.transform(valid_df.label)
    print(f"valid: {len(X_valid)}")
    del valid_df
    gc.collect()

    n_input = 768
    n_hidden = 128
    n_output = 6
    model = NNModel(n_input, n_hidden, n_output)
    model = try_gpu(model)
    print("model defined")

    executer = TrainValidExecuter()
    train_loss = executer.execute(model, X_train, y_train, X_valid, y_valid)
    print(f"train loss: {train_loss}")
    
    print("DONE")


if __name__ == "__main__":
    main()
