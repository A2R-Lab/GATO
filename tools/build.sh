#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the build directory
BUILD_DIR="build"

# Clean the build directory
echo "Cleaning build directory: $BUILD_DIR"
rm -rf "$BUILD_DIR"

# Recreate the build directory and navigate into it
echo "Creating build directory: $BUILD_DIR"
mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

# Run CMake to configure the project
echo "Configuring project with CMake..."
cmake ..

# Build the project
echo "Building project..."
cmake --build . --parallel

echo "Build complete!"
