# generate schema by using openapi-generator

## setup

```shell
npm install @openapitools/openapi-generator-cli -g
```

## generate

```shell
npx @openapitools/openapi-generator-cli generate -i openapi.yaml -g python-fastapi -o ./app
```

## run

```shell
cd app/src
uvicorn openapi_server.main:app --host 0.0.0.0 --port 8080
```
