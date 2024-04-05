# use nvidia docker

## run

```shell
docker run --gpus all -it --rm -v ./sample:/workspace/sample nvcr.io/nvidia/pytorch:24.03-py3
```

```shell
cd sample
python -B main.py
```

reference

- https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
- https://docs.nvidia.com/deeplearning/frameworks/pdf/PyTorch-Release-Notes.pdf
