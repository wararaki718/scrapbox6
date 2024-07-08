import numpy as np
from fastembed import TextEmbedding
from torchtext.datasets import AG_NEWS
from torch.utils.data.datapipes.iter.sharding import ShardingFilterIterDataPipe

from model import NNModel


def main() -> None:
    train, test = AG_NEWS()

    model_name = "BAAI/bge-small-en-v1.5"
    model: TextEmbedding = TextEmbedding(
        model_name=model_name,
        providers=["CUDAExecutionProvider"],
    )

    for label, sentence in train:
        print(label)
        print(sentence)
        embedding: np.ndarray = next(iter(model.embed(sentence)))
        print(embedding.shape)
        print(type(embedding))
        break
    print("DONE")


if __name__ == "__main__":
    main()
