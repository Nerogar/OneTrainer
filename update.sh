#!/usr/bin/env bash

set -e

# Change our working dir to the root of the project.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Pull the latest changes via Git.
echo "[OneTrainer] Updating OneTrainer to latest version from Git repository..."
git pull

# Load the newest version of the function library.
source "lib.include.sh"


if [[ -f "last-commit.sha" ]]; then
    last_commit=$(<"last-commit.sha")
    current_commit=$(git rev-parse HEAD)
    if [[ "$last_commit" == "$current_commit" && "$1" != "force" ]]; then
        echo "Skipping dependency check. Run \"./update.sh force\" to check anyway."
        exit 0
    fi
fi

# Prepare runtime and upgrade all dependencies to latest compatible version.
prepare_runtime_environment upgrade

git rev-parse HEAD > last-commit.sha
