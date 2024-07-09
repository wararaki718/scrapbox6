import gc

import numpy as np
import torch
from torchtext.datasets import AG_NEWS
from sklearn.feature_extraction.text import TfidfVectorizer

from model import NNModel
from trainer import Trainer
from utils import try_gpu


def main() -> None:
    train, test = AG_NEWS()

    labels = []
    sentences = []
    for label, sentence in train:
        labels.append(label)
        sentences.append(sentence)

    labels = np.array(labels) - 1 # scaling 
    print(np.unique(labels))

    sentences = np.array(sentences)
    print(f"n_data: {len(sentences)}")
    # print(sentences[:3])
    print()

    # sorted
    indices = np.argsort(labels)
    labels = labels[indices]
    sentences = sentences[indices]
    
    # vectorized
    vectorizer = TfidfVectorizer(max_df=0.99, min_df=0.01)
    X_tfidf = vectorizer.fit_transform(sentences)
    X = torch.Tensor(X_tfidf.toarray())
    y = torch.Tensor(labels)
    print(f"vocabs: {len(vectorizer.get_feature_names_out())}")
    print()
    del X_tfidf, label, vectorizer, sentences
    gc.collect()
    print(X.shape)
    print(y.shape)

    # model
    model = NNModel(
        n_input=X.shape[1],
        n_output=len(set(labels)),
    )
    model = try_gpu(model)
    trainer = Trainer()

    print("DONE")


if __name__ == "__main__":
    main()
