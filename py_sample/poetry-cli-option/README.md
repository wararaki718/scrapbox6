# cli option check

## setup

```shell
poetry install
```

## run

```shell
poetry run cliapp
```

```shell
poetry run cliapp --name custom
```

show help

```shell
poetry run cliapp --help
```

## use docker

build

```shell
docker-compose build
```

run

```shell
docker-compose -f docker-compose.yml up
```

custom

```shell
docker-compose -f docker-compose.yml -f docker-compose.custom.yml up
```
