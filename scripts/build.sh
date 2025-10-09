#!/bin/bash

# check if docker is available if not say you are probably running this inside a docker container
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. You are probably running this inside a docker container."
    echo "Run it on the host machine instead."
    exit 1
fi

docker buildx build --platform linux/arm64 -f Dockerfile -t maxscha/research-environment:latest .
if [ $? -ne 0 ]; then
    echo "ARM build failed"
    exit
fi

docker buildx build --platform linux/amd64 -f Dockerfile -t maxscha/research-environment:latest .
if [ $? -ne 0 ]; then
    echo "AMD build failed"
    exit
fi

docker buildx build --platform linux/amd64,linux/arm64 -f Dockerfile -t maxscha/research-environment:latest --push .