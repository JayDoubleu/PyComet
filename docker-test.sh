#!/bin/bash
set -e

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_section() {
    echo -e "\n${BLUE}===============================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${BLUE}===============================================${NC}\n"
}

# Build the Docker image
print_section "Building Docker test image..."
docker build -f Dockerfile.test -t pycomet-test .

# Run specific tests based on arguments
if [ "$1" == "hooks" ]; then
    print_section "Running Git Hooks tests..."
    docker run --rm -v "$(pwd):/app" pycomet-test pytest tests/test_git.py tests/test_cli_hooks.py -v
elif [ "$1" == "all" ]; then
    print_section "Running all tests (excluding integration tests)..."
    docker run --rm -v "$(pwd):/app" pycomet-test pytest tests/ -v -m "not integration"
else
    print_section "Running tests with custom command..."
    docker run --rm -v "$(pwd):/app" pycomet-test pytest "$@"
fi