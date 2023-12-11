# use fiarseq

## download model

```shell
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz
tar xvzf bart.large.mnli.tar.gz
```

## setup

```shell
pip install torch fairseq protobuf==3.20.3
```

## run

```shell
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

```shell
python main.py
```
