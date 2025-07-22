#!/bin/bash
set -e

# Activate virtual environment
source /opt/venv/bin/activate

# Check for NVIDIA GPU
if ! nvidia-smi &> /dev/null; then
    echo "WARNING: No NVIDIA GPU detected or NVIDIA drivers not properly installed"
fi

# Verify Python version
python --version

# Run the command
exec "$@"
