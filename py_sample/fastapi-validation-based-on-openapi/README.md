# fastapi validation based on openapi

## setup

```shell
pip install datamodel-code-generator uvicorn fastapi
```

## generate schema

```shell
datamodel-codegen --input interface/openapi.yaml --input-file-type openapi --output api/schema/model.py
```

## run

```shell
uvicorn api.main:app --reload
```

open http://localhost:8000/docs on your browser.
