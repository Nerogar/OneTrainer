#!/usr/bin/env bash

set -e

if [ "$OT_PIP_INSTALL" = "true" ]; then
    echo "[OneTrainer] Installing dependencies via pip..."
    bash scripts/pip-install/install.sh
    exit 0
fi

source "${BASH_SOURCE[0]%/*}/lib.include.sh"

prepare_runtime_environment
