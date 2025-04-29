#!/usr/bin/env bash
set -e

# Get list of Python files to process (excluding test files)
list_of_files=$(git ls-files | grep -E '\.py$' | grep -v '^tests/')

# Function to handle in-place formatting confirmation
confirm_in_place() {
    while true; do
        echo "Remember to save files before running formatting!"
        read -p "Ready to run formatting in-place? [Yn]:" yn
        case $yn in
            [Nn]* ) exit 1;;
            * ) break;;
        esac
    done
}

# Run formatting fixes
echo "Running Ruff formatting"
confirm_in_place
ruff format $list_of_files
echo "Running Ruff autofix for lint issues"
ruff check --fix $list_of_files

echo "Formatting completed successfully!"
