# llm chat api

## setup

```shell
pip install fastapi uvicorn transformers
```

## run

```shell
uvicorn main:app
```

open http://localhost:8000/docs on your browser.

sample question

```json
{
  "text": "Where is the capital of Japan?"
}
```
