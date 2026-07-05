#!/usr/bin/env bash

set -e

source "${BASH_SOURCE[0]%/*}/lib.include.sh"

for arg in "$@"; do
    if [[ "$arg" == "--dev" ]]; then
        export OT_DEV=1
    fi
done

# Xet is buggy - https://github.com/Nerogar/OneTrainer/issues/949
if [[ -z "${HF_HUB_DISABLE_XET+x}" ]]; then
    export HF_HUB_DISABLE_XET=1
fi

prepare_runtime_environment

if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed or not in PATH"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$SCRIPT_DIR/web/gui/dist/main/main/index.cjs" ]]; then
    echo "Error: Web GUI has not been built yet."
    echo "Please run install.sh or update.sh first to build the web UI."
    exit 1
fi

echo "Starting OneTrainer Web UI..."
cd "$SCRIPT_DIR/web/gui"
npx electron .
