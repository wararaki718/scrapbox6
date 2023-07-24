#!/bin/bash

echo ""
curl -XPUT -H "Content-Type: application/json" localhost:6333/collections/sample/points -d '@data/data.json'
echo ""
echo ""
