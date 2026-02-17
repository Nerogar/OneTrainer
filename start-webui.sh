#!/usr/bin/env bash

set -e

source "${BASH_SOURCE[0]%/*}/lib.include.sh"

# Xet is buggy. Disabled by default unless already defined
if [[ -z "${HF_HUB_DISABLE_XET+x}" ]]; then
    export HF_HUB_DISABLE_XET=1
fi

prepare_runtime_environment

WEBUI_HOST="${WEBUI_HOST:-127.0.0.1}"
WEBUI_PORT="${WEBUI_PORT:-7860}"

run_python_in_active_env "scripts/train_webui.py" --host "${WEBUI_HOST}" --port "${WEBUI_PORT}" "$@"
