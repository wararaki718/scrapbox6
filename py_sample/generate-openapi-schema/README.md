# schema generate for fastapi

## setup

```shell
pip install datamodel-code-generator
```

## generate schema

```shell
datamodel-codegen --input openapi.yaml --input-file-type openapi --output app/schema/model.py
```

## run

```shell
python app/main.py
```

access to http://localhost:8000/docs
