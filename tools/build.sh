#!/bin/bash

# need to run: chmod +x tools/build.sh

# Exit on any error
set -e

if [ "$1" == "clean" ]; then
    ./tools/cleanup.sh
    exit 0
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored output
print_color() {
    printf "${2}${1}${NC}\n"
}

# Get the root directory of the project
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Create and enter build directory
print_color "Creating build directory..." "$YELLOW"
mkdir -p "${ROOT_DIR}/build"
cd "${ROOT_DIR}/build"
mkdir -p "results"

# Run CMake
print_color "Running CMake..." "$YELLOW"
cmake ..

# Build the project
print_color "Building project..." "$YELLOW"
make -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    print_color "Build successful!" "$GREEN"
else
    print_color "Build failed." "$RED"
    exit 1
fi

print_color "Build script completed successfully!" "$GREEN"