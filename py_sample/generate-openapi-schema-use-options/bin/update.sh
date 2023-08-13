#!/bin/bash

# update the schema
datamodel-codegen --input openapi.yaml --input-file-type openapi --output api/schema.py

echo "DONE"