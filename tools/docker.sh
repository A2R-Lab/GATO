#!/bin/bash

# Source common definitions
source "$(dirname "$0")/common.sh"

printf "\n${CYAN}${BOLD}--------------------------------------------------${RESET}\n"
printf "${BOLD}${GREEN}${GEAR} Setting up docker container.${RESET}\n"

IMAGE_NAME="gato"
# Check if the image exists
if ! docker image inspect ${IMAGE_NAME} >/dev/null 2>&1; then
    printf "${YELLOW}${BOLD}${GEAR} Image '${IMAGE_NAME}' not found. Building...${RESET}\n\n"
    docker build -t ${IMAGE_NAME} .
    if [ $? -ne 0 ]; then
        printf "\n${RED}${BOLD}${CROSS} Build failed. Exiting.${RESET}\n"
        exit 1
    fi
    printf "\n${GREEN}${BOLD}${CHECK} Image '${IMAGE_NAME}' built successfully.${RESET}\n"
else
    printf "${GREEN}${BOLD}${CHECK} Image '${IMAGE_NAME}' found.${RESET}\n"
fi

CONTAINER_NAME="gato-container"
# printf "${CYAN}${BOLD}--------------------------------------------------${RESET}\n\n"

# Prepare for GUI forwarding if needed (outside container check)
export DISPLAY=${DISPLAY:-:0} # Default to :0 if not set
xhost +local:docker >/dev/null 2>&1 # Silence output, might fail if no X server

EXIT_CODE=0

# Check if container exists and is running
if docker ps -q -f name=^/${CONTAINER_NAME}$ | grep -q .; then
    printf "${BOLD}${GREEN}${ARROW} Attaching to running container ${YELLOW}${CONTAINER_NAME}${RESET}...\n\n"
    docker exec -it $CONTAINER_NAME /bin/bash
    EXIT_CODE=$?
# Check if container exists but is stopped
elif docker ps -aq -f status=exited -f name=^/${CONTAINER_NAME}$ | grep -q .; then
    printf "${YELLOW}${BOLD}${GEAR} Starting stopped container ${YELLOW}${CONTAINER_NAME}${RESET} and attaching...\n\n"
    docker start $CONTAINER_NAME >/dev/null
    docker exec -it $CONTAINER_NAME /bin/bash
    EXIT_CODE=$?
# Container does not exist, run a new one
else
    printf "${BOLD}${GREEN}${ARROW} Running new container (name: ${YELLOW}${CONTAINER_NAME}${YELLOW})...${RESET}\n\n"
    docker run -it \
        --gpus all \
        --network=host \
        -e DISPLAY=:0 \
        -v "$(pwd)":/workspace \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        --name ${CONTAINER_NAME} \
        ${IMAGE_NAME}
    EXIT_CODE=$?
fi

if [ "$EXIT_CODE" -eq 0 ]; then
    printf "\n${GREEN}${BOLD}${CHECK} Exited container session for ${YELLOW}${CONTAINER_NAME}${RESET} successfully.${RESET}\n"
else
    printf "\n${RED}${BOLD}${CROSS} Command failed or container exited with error (Code: $EXIT_CODE).${RESET}\n"
fi
printf "${CYAN}${BOLD}--------------------------------------------------${RESET}\n\n"

# docker build -t gato .
# docker run --gpus all -it --name mycontainer -v $(pwd):/workspace 
