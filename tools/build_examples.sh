#!/bin/bash

# Exit on any error
set -e

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

# Create and enter build directory for examples
print_color "Creating build directory for examples..." "$YELLOW"
mkdir -p "${ROOT_DIR}/examples/build"
cd "${ROOT_DIR}/examples/build"

# Run CMake
print_color "Running CMake for examples..." "$YELLOW"
cmake ..

# Build the project
print_color "Building examples..." "$YELLOW"
make -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    print_color "Examples build successful!" "$GREEN"
else
    print_color "Examples build failed." "$RED"
    exit 1
fi

print_color "Examples build script completed successfully!" "$GREEN"