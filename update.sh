#!/usr/bin/env bash

set -e

# Change our working dir to the root of the project.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Pull the latest changes via Git.
echo "[OneTrainer] Updating OneTrainer to latest version from Git repository..."
git pull

# Load the newest version of the function library.
source "lib.include.sh"

# Prepare runtime and upgrade all dependencies to latest compatible version.
prepare_runtime_environment upgrade
