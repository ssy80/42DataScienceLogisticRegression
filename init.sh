#!/bin/bash 

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt

# Enter into the virtual environment
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    exec "$SHELL"
fi

# To check if the virtual environment is activated
which python

# To make sure tkinter is installed
python3 -c "import tkinter; print('tkinter OK')"
python3 -c "import matplotlib; import sys; print(matplotlib.get_backend(), sys.executable)"