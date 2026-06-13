#!/usr/bin/env bash

set -e

# Xet is buggy. Disabled by default unless already defined - https://github.com/Nerogar/OneTrainer/issues/949
if [[ -z "${HF_HUB_DISABLE_XET+x}" ]]; then
    export HF_HUB_DISABLE_XET=1
fi

source "${BASH_SOURCE[0]%/*}/lib.include.sh"

# Fetch and validate the name of the target script.
if [[ -z "${1}" ]]; then
    print_error "You must provide the name of the script to execute, such as \"train\"."
    exit 1
fi

OT_CUSTOM_SCRIPT_FILE="scripts/${1}.py"
if [[ ! -f "${OT_CUSTOM_SCRIPT_FILE}" ]]; then
    print_error "Custom script file \"${OT_CUSTOM_SCRIPT_FILE}\" does not exist."
    exit 1
fi

prepare_runtime_environment

# Remove $1 (name of the script) and pass all remaining arguments to the script.
shift
run_python_in_active_env "${OT_CUSTOM_SCRIPT_FILE}" "$@"
