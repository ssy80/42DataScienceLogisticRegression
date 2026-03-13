#!/bin/bash 

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Enter into the virtual environment
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    exec "$SHELL"
fi

# To check if the virtual environment is activated, you can run:
which python