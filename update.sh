#!/usr/bin/env bash

set -eo pipefail

if [ "$OT_PIP_INSTALL" = "true" ]; then
    echo "[OneTrainer] Updating dependencies via pip..."
    bash scripts/pip-install/update.sh
    exit 0
fi

# Change our working dir to the root of the project.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Pull the latest changes via Git.
echo "[OneTrainer] Updating OneTrainer to latest version from Git repository..."

if [ ! -d .git ]; then
    git init
    git remote add origin https://github.com/Nerogar/OneTrainer.git
    git fetch origin master
    git switch -f -C master --track origin/master
fi

git pull

# Load the newest version of the function library.
source "lib.include.sh"

# Prepare runtime and upgrade all dependencies to latest compatible version.
prepare_runtime_environment upgrade
