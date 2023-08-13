# use options

## setup

```shell
pip install datamodel-code-generator fastapi-code-generator fastapi uvicorn
```

## generate schemas

```shell
bash ./bin/generate.sh
```

## update schemas

```shell
bash ./bin/update.sh
```

## run

```shell
uvicorn api.main:app
```

access to http://localhost:8000/docs
