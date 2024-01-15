# vector search

## build & run

use intellij

## launch opensearch

```shell
docker-compose up
```

open http://localhost:5601 on your browser.

## check the behaviors

search

```shell
curl -XPOST -H "Content-Type: application/json" 'http://localhost:8080/vector/search -d "{\"vector\": [2.0, 3.0]}"
```
