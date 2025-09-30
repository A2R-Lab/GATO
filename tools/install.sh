#!/bin/bash

git submodule update --init --recursive

echo -e "----------------------------------------"
echo -e "Installing dependencies...\n"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv sync
source .venv/bin/activate

echo -e "----------------------------------------"
echo -e "Building docker image..."
IMAGE_NAME="gato"
CONTAINER_NAME="gato-container"
docker build -t ${IMAGE_NAME} .

echo -e "----------------------------------------"
echo -e "Ensuring container is running..."
if docker ps -q -f name=^/${CONTAINER_NAME}$ | grep -q .; then
    echo -e "Container '${CONTAINER_NAME}' already running."
elif docker ps -aq -f name=^/${CONTAINER_NAME}$ | grep -q .; then
    docker start ${CONTAINER_NAME}
else
    docker run -d -it \
        --gpus all \
        --network=host \
        -e DISPLAY=${DISPLAY:-:0} \
        -v "$(pwd)":/workspace \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        --name ${CONTAINER_NAME} \
        ${IMAGE_NAME}
fi

echo -e "----------------------------------------"
echo -e "Building ..."
docker exec ${CONTAINER_NAME} bash -c "cd /workspace && make build"

echo -e "----------------------------------------"
echo -e "Setup complete."
echo -e " - to enter the container: 'docker exec -it ${CONTAINER_NAME} bash'"
echo -e " - to use the venv: 'source .venv/bin/activate'"
#docker exec -it ${CONTAINER_NAME} bash