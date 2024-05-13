# delete behavior check

## setup

```shell
pip install qdrant-client==1.3.0 transformers scikit-learn
```

## run

launch qdrant

```shell
docker-compose up
```

```shell
python main.py
```

delete

```shell
python delete_collection.py
```

```shell
python delete_points.py
```
