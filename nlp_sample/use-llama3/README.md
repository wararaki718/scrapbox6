# use llama3

## download model

accept meta's license

- https://llama.meta.com/llama-downloads/

see read.me

- https://github.com/meta-llama/llama3/blob/main/README.md
  - copy url & execute download.sh (enter the url & model)

## setup

```shell
pip install git+https://github.com/meta-llama/llama3.git
```

## run

```shell
torchrun --nproc_per_node 1 main.py
```
