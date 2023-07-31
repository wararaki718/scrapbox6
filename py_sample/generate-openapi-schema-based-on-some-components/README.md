# generate openapi schema

## setup

```shell
pip install datamodel-code-generator
```

## generate schema

```shell
datamodel-codegen --input docs/openapi.yaml --input-file-type openapi --output app/schema/model.py
```

## run

```shell
python app/main.py
```
