#!/bin/bash

# need to run: chmod +x tools/cleanup.sh

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

# Function to remove a directory if it exists
remove_dir_if_exists() {
    if [ -d "$1" ]; then
        print_color "Removing $1..." "$YELLOW"
        rm -rf "$1"
    fi
}

# Clean up build directory
BUILD_DIR="${ROOT_DIR}/build"
if [ -d "$BUILD_DIR" ]; then
    print_color "Cleaning up build directory..." "$YELLOW"
    
    # Run 'make clean' if Makefile exists
    if [ -f "${BUILD_DIR}/Makefile" ]; then
        print_color "Running 'make clean'..." "$YELLOW"
        cd "$BUILD_DIR"
        make clean
        cd "$ROOT_DIR"
    fi
    
    # Remove build directory
    remove_dir_if_exists "$BUILD_DIR"
else
    print_color "Build directory does not exist. Nothing to clean." "$GREEN"
fi

# Clean up CMake cache files
print_color "Removing CMake cache files..." "$YELLOW"
find "$ROOT_DIR" -name CMakeCache.txt -delete
find "$ROOT_DIR" -name CMakeFiles -type d -exec rm -rf {} +

# Clean up any other build artifacts or temporary files
# Add more cleanup commands here if needed
# For example:
# remove_dir_if_exists "${ROOT_DIR}/examples/build"
# remove_dir_if_exists "${ROOT_DIR}/experiments/MPCGPU/build"

print_color "Cleanup completed successfully!" "$GREEN"