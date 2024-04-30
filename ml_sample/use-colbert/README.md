# colbert

## setup

```shell
pip install colbert-ai
```

## download

```shell
wget -O data/collections.tsv https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/collections/japanese_collection.tsv
```

```shell
wget -O data/queries.dev.small.tsv https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/japanese_queries.dev.small.tsv
```

```shell
wget -O data/qrels.dev.small.tsv https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/qrels.dev.small.tsv
```

```shell
wget -O data/triplets.train.ids.small.tsv https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/triples.train.ids.small.tsv
```

## run

```shell
python main.py
```
