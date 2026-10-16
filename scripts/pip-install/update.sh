#!/usr/bin/env bash

set -e

# Pull the latest changes via Git.
echo "[OneTrainer] Updating OneTrainer to latest version from Git repository..."
git pull

# Load the newest version of the function library.
source "${BASH_SOURCE[0]%/*}/lib.include.sh"

# Prepare runtime and upgrade all dependencies to latest compatible version.
prepare_runtime_environment upgrade
