# use opensearch with springboot

## build & run

use intellij

## launch opensearch

```shell
docker-compose up
```

open http://localhost:5601 on your browser.

## check the behaviors

insert data

```shell
curl -XPOST 'http://localhost:8080/marketplace/insert?name=username'
```

search

```shell
curl 'http://localhost:8080/marketplace/search'
```

```shell
curl 'http://localhost:8080/marketplace/search?name=username'
```
