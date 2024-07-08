import gc

import torch
from fastembed import TextEmbedding
from torchtext.datasets import AG_NEWS

from model import NNModel


def main() -> None:
    train, test = AG_NEWS()

    # TODO: change embedding tool
    model_name = "BAAI/bge-small-en-v1.5"
    model: TextEmbedding = TextEmbedding(
        model_name=model_name,
        providers=["CUDAExecutionProvider"],
    )

    labels = []
    sentences = []
    for label, sentence in train:
        labels.append(label)
        sentences.append(sentence)

    y = torch.Tensor(labels)
    X: torch.Tensor = torch.cat(
        tuple(map(lambda x: torch.Tensor(x), model.embed(sentences))),
        dim=0,
    )
    print(y.shape)
    print(X.shape)
    del sentences
    gc.collect()

    print("DONE")


if __name__ == "__main__":
    main()
