#!/usr/bin/env bash
set -e

# Get list of Python files to process (excluding test files)
list_of_files=$(git ls-files | grep -E '\.py$' | grep -v '^tests/')

# Run formatting checks
echo "Running Ruff for formatting checks"
ruff format --check $list_of_files
if [ $? -ne 0 ]; then
    echo "Ruff formatting checks did not pass"
    exit 2
fi
echo "Ruff formatting checks passed!"

echo "All checks completed successfully!"
