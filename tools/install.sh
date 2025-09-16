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
echo -e "Starting container..."
docker compose up -d

echo -e "----------------------------------------"
echo -e "Building ..."
docker compose exec dev bash -c "make build"

echo -e "----------------------------------------"
echo -e "Setup complete."
echo -e " - to enter the container: 'docker compose exec dev bash'"
echo -e " - to use the venv: 'source .venv/bin/activate'"

#docker compose exec dev bash