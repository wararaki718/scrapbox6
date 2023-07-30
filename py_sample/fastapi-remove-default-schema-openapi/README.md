# remove default schema

## setup

```shell
pip install fastapi-code-generator uvicorn
```

## create schema

```shell
fastapi-codegen --input openapi.yaml --output api --model-file schema
```

## run

```shell
uvicorn api.main:app --reload
```
