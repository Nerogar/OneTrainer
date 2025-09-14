#!/usr/bin/env bash

set -e

source "${BASH_SOURCE[0]%/*}/lib.include.sh"

# Xet is buggy. Disabled by default unless already defined - https://github.com/Nerogar/OneTrainer/issues/949
if [[ -z "${HF_HUB_DISABLE_XET+x}" ]]; then
    export HF_HUB_DISABLE_XET=1
fi

prepare_runtime_environment

run_python_in_active_env "scripts/train_ui.py" "$@"
