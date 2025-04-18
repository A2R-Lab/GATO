#!/bin/bash

if ! docker image inspect gato-env &>/dev/null; then
    echo "Building Docker image..."
    docker build -t gato-env .
fi

docker run -it \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    --network host \
    --rm \
    -v $(pwd):/workspace \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    gato-env