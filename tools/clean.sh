#! /bin/bash

# Source common definitions
source "$(dirname "$0")/common.sh"

printf "\n${CYAN}${BOLD}--------------------------------------------------${RESET}\n"

printf "${ARROW} Removing build directories...\n"
rm -rf build bindings/build 2>/dev/null || sudo rm -rf build bindings/build

if docker ps -q -f name=^/gato-container$ | grep -q .; then
    printf "${ARROW} Stopping running container '${YELLOW}gato-container${RESET}'...\n"
    docker stop gato-container > /dev/null
fi

# Remove container if it exists (stopped or running)
if docker ps -aq -f name=^/gato-container$ | grep -q .; then
    printf "${ARROW} Removing container '${YELLOW}gato-container${RESET}'...\n"
    docker rm -f gato-container > /dev/null
else
    printf "${YELLOW}${BOLD}i${RESET} Container '${YELLOW}gato-container${RESET}' not found, skipping removal.\n"
fi

# Remove image if it exists
if docker image inspect gato >/dev/null 2>&1; then
    printf "${ARROW} Removing image '${YELLOW}gato${RESET}'...\n"
    docker rmi gato > /dev/null
else
    printf "${YELLOW}${BOLD}i${RESET} Image '${YELLOW}gato${RESET}' not found, skipping removal.\n"
fi

printf "${BOLD}${GREEN}${CHECK} Successfully cleaned up.${RESET}\n"
printf "${CYAN}${BOLD}--------------------------------------------------${RESET}\n\n"
