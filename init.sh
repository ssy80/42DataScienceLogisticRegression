#!/bin/bash 

# Install system dependencies for Matplotlib GUI backend
sudo apt update
sudo apt install -y python3-tk

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt

# Enter into the virtual environment
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    exec "$SHELL"
fi

# Verify environment
echo "--- Environment Check ---"
which python
python -c "import tkinter; print('Tkinter: OK')"
python -c "import matplotlib; print('Matplotlib Backend: ' + matplotlib.get_backend())"