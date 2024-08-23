#!/bin/bash

# need to run: chmod +x tools/bindings.sh

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

# Get the bindings directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINDINGS_DIR="${ROOT_DIR}/bindings/python"

# Create and enter build directory
print_color "Creating build directory..." "$YELLOW"
mkdir -p "${BINDINGS_DIR}/build"
cd "${BINDINGS_DIR}/build"

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

# Create a shell script to set up the environment
ENV_SETUP_SCRIPT="${ROOT_DIR}/tools/setup_gato_env.sh"
cat > "${ENV_SETUP_SCRIPT}" << EOL
#!/bin/bash
export PYTHONPATH="\${PYTHONPATH}:${BINDINGS_DIR}/build"
echo "GATO environment set up. You can now use the library in Python."
EOL

chmod +x "${ENV_SETUP_SCRIPT}"

print_color "Environment setup script created at: ${ENV_SETUP_SCRIPT}" "$GREEN"

print_color "\nTo use GATO, run: source ${ENV_SETUP_SCRIPT}\n" "$NC"

print_color "Build script completed successfully!" "$GREEN"