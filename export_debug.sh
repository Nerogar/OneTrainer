#!/bin/bash
# filepath: /c:/repos/OneTrainer/export_debug.sh
set -euo pipefail

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/venv"

# Check if virtual environment directory exists
if [ ! -d "$VENV_DIR" ]; then
  echo "venv not found, please run install.sh first."
  exit 1
fi

echo "Activating venv at $VENV_DIR..."
# Activate the virtual environment
source "$VENV_DIR/bin/activate"

echo "Using Python: $(python --version)"

# Run the debug report generation script
python "$SCRIPT_DIR/scripts/generate_debug_report.py"

echo "Debug report generation complete."
read -p "Press [Enter] to exit..."
