#!/bin/bash

# Deactivate the virtual environment
# This only works if the script is executed with 'source remove.sh'
if type deactivate &>/dev/null; then
    deactivate
    echo "Virtual environment deactivated."
else
    echo "Note: No active virtual environment found in this shell."
fi

# Remove the virtual environment directory 
if [ -d "venv" ]; then
    rm -rf venv
    echo "Removed 'venv' directory."
elif [ -d ".venv" ]; then
    rm -rf .venv
    echo "Removed '.venv' directory."
fi

# Remove generated output files from the logistic regression scripts
# Checking root, mandatory, and bonus directories
FILES_TO_REMOVE=(
    "weights.csv" "houses.csv"
    "mandatory/weights.csv" "mandatory/houses.csv"
    "bonus/weights.csv" "bonus/houses.csv"
)

for file in "${FILES_TO_REMOVE[@]}"; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "Removed '$file'."
    fi
done

# Clean up Python cache files recursively
find . -type d -name "__pycache__" -exec rm -rf {} +
echo "Removed __pycache__ directories."

echo "Cleanup complete!"