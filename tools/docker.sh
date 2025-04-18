#!/bin/bash

echo -e "----------------------------------------"
echo -e "Starting container...\n"
docker compose --parallel=-1 --project-name=gato up -d 

echo -e "----------------------------------------"
echo -e "Entering container...\n"
docker compose exec dev bash


# old script from emre

# if ! docker image inspect gato-env &>/dev/null; then
#     echo "Building Docker image..."
#     docker build -t gato-env .
# fi

# docker run -it \
#     --gpus all \
#     -e DISPLAY=$DISPLAY \
#     --network host \
#     --rm \
#     -v $(pwd):/workspace \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     gato-env


# ------
# docker build -t gato .
# docker run --gpus all -it -v $(pwd):/app gato