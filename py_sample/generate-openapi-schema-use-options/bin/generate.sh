#!/bin/bash

# create an init application
fastapi-codegen --input openapi.yaml --output api --model-file schema.py

echo "DONE"
