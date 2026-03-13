#!/bin/bash

# 1. Deactivate the virtual environment
# This only works if the script is executed with 'source remove.sh'
if type deactivate &>/dev/null; then
    deactivate
    echo "Virtual environment deactivated."
else
    echo "Note: No active virtual environment found in this shell."
fi

# 2. Remove the virtual environment directory 
# (Checking for both 'venv' and '.venv' which are the most common naming conventions)
if [ -d "venv" ]; then
    rm -rf venv
    echo "Removed 'venv' directory."
elif [ -d ".venv" ]; then
    rm -rf .venv
    echo "Removed '.venv' directory."

fi

# 3. Remove generated output files from the logistic regression scripts
if [ -f "weights.csv" ]; then
    rm -f weights.csv
    echo "Removed 'weights.csv'."
fi

if [ -f "houses.csv" ]; then
    rm -f houses.csv
    echo "Removed 'houses.csv'."
fi

# 4. Clean up Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} +
echo "Removed __pycache__ directories."

echo "Cleanup complete!"