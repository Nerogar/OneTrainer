#!/usr/bin/env bash

set -eo pipefail

if [ "$OT_PIP_INSTALL" = "true" ]; then
    echo "[OneTrainer] Running UI inside virtual environment..."
    bash ./scripts/pip-install/start-ui.sh
    exit 0
fi

source "${BASH_SOURCE[0]%/*}/lib.include.sh"

# Xet is buggy. Disabled by default unless already defined - https://github.com/Nerogar/OneTrainer/issues/949
if [[ -z "${HF_HUB_DISABLE_XET+x}" ]]; then
    export HF_HUB_DISABLE_XET=1
fi

prepare_runtime_environment

run_in_env python "scripts/train_ui.py" "$@"
