#!/usr/bin/env bash

set -e

# Detect absolute path to the directory where "lib.include.sh" resides.
export SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Guard against including the library multiple times.
readonly SCRIPT_DIR

# Ensure that all scripts change their working dir to the root of the project.
cd -- "${SCRIPT_DIR}"

# User-configurable environment variables.
# IMPORTANT: Don't modify the code below! Pass these variables via the environment!
export OT_CUDA_LOWMEM_MODE="${OT_CUDA_LOWMEM_MODE:-false}"
export OT_PLATFORM="${OT_PLATFORM:-detect}"
export OT_SCRIPT_DEBUG="${OT_SCRIPT_DEBUG:-false}"

# Internal environment variables.
# NOTE: Version check supports "3", "3.1" and "3.1.5" specifier formats.
export OT_PYTHON_VERSION_MINIMUM="3.10"
export OT_PYTHON_VERSION_TOO_HIGH="3.14"
export OT_UPDATE_METADATA_FILE="${SCRIPT_DIR}/update.var"
export OT_HOST_OS="$(uname -s)"

# Force PyTorch to use fallbacks on Mac systems.
if [[ "${OT_HOST_OS}" == "Darwin" ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK="1"
fi

# Change PyTorch memory allocation to reduce CUDA out-of-memory situations.
if [[ "${OT_CUDA_LOWMEM_MODE}" == "true" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"
fi

# Utility functions.
function escape_shell_command {
    # NOTE: "%q" ensures shell-compatible argument escaping.
    printf " %q" "$@" | sed 's/^ //'
}

function print {
    # NOTE: "%b" parses escape-sequences, allowing us to output "\n" newlines.
    printf "[OneTrainer] %b\n" "$*"
}

function print_warning {
    printf "[OneTrainer] Warning: %b\n" "$*" >&2
}

function print_error {
    printf "[OneTrainer] Error: %b\n" "$*" >&2
}

function print_debug {
    if [[ "${OT_SCRIPT_DEBUG}" == "true" ]]; then
        print "Debug: $*"
    fi
}

function print_command {
    # NOTE: "%s" prints the escaped command as-is without parsing escape-seqs.
    printf "[OneTrainer] + %s\n" "$(escape_shell_command "$@")"
}

# Retrieves the most recent Git commit's hash.
function get_current_git_hash {
    # NOTE: Will not detect local code changes, only the newest commit's hash.
    git rev-parse HEAD
}

# Writes update-check metadata to disk.
function save_update_metadata {
    get_current_git_hash >"${OT_UPDATE_METADATA_FILE}"
}

# Checks whether the current Git state matches the last-seen metadata state.
function is_update_metadata_changed {
    if [[ -f "${OT_UPDATE_METADATA_FILE}" ]]; then
        local saved_hash="$(<"${OT_UPDATE_METADATA_FILE}")"
        local current_hash="$(get_current_git_hash)"
        print_debug "Saved Metadata Hash=\"${saved_hash}\", Current Hash=\"${current_hash}\""
        if [[ "${saved_hash}" == "${current_hash}" ]]; then
            # Signal "failure", meaning "metadata is NOT outdated, abort".
            return 1
        fi
    fi

    return 0
}

# Escapes all special regex characters in string.
function regex_escape {
    sed 's/[][\.|$(){}?+*^]/\\&/g' <<<"$*"
}

# Resolves the absolute path for an absolute or relative input path.
function absolute_path {
    if [[ -z "$1" ]]; then
        print_error "absolute_path requires 1 argument."
        return 1
    fi

    if [[ ! -d "$1" ]]; then
        print_error "absolute_path argument is not a directory: \"$1\"."
        return 1
    fi

    echo "$(cd -- "$1" &>/dev/null && pwd)"
}

# Checks if a command exists and is executable.
function can_exec {
    if [[ -z "$1" ]]; then
        print_error "can_exec requires 1 argument."
        return 1
    fi

    if local full_path="$(command -v "$1" 2>/dev/null)"; then
        if [[ ! -z "${full_path}" ]] && [[ -x "${full_path}" ]]; then
            return 0
        fi
    fi

    return 1
}

# Executes a shell command and displays the exact command for logging purposes.
function run_cmd {
    print_command "$@"
    "$@"
}

function get_platform {
    # NOTE: The user can override our platform detection via the environment.
    local platform="${OT_PLATFORM}"
    if [[ "${platform}" == "detect" ]]; then
        # NOTE: We MUST prioritize NVIDIA first, since machines that contain
        # *both* AMD and NVIDIA GPUs are usually running integrated AMD graphics
        # that's built into their CPU, whereas their *dedicated* GPU is NVIDIA.
        if [[ -e "/dev/nvidia0" ]] || can_exec nvidia-smi || can_exec "/usr/lib/wsl/lib/nvidia-smi"; then
            # NVIDIA graphics.
            #  "/dev/nvidia0": The "first" detected NVIDIA GPU in the system.
            #  "nvidia-smi": Driver tool for all NVIDIA GPUs made after 2010.
            #  "nvcc": CUDA SDK compiler. Not included in the drivers.
            #  "/usr/lib/wsl/lib/nvidia-smi": WSL's NVIDIA path (isn't in $PATH).
            # SEE: https://docs.nvidia.com/cuda/wsl-user-guide/
            platform="cuda"
        elif [[ -e "/dev/kfd" ]]; then
            # AMD graphics.
            platform="rocm"
        else
            # No GPU acceleration.
            platform="cpu"
        fi
    fi

    echo "${platform}"
}

function install_env {
    run_cmd pixi install --locked -e "${OT_PLATFORM}"
}

function run_in_env {
    run_cmd pixi run --locked -e "${OT_PLATFORM}" "$@"
}

function get_or_update_pixi {
    if can_exec pixi; then
        print_debug '`pixi` found, updating.'
        run_cmd pixi self-update
    else
        print_debug '`pixi` not found, attempting installation.'
        if can_exec curl; then
            print_debug '`curl` found, installing `pixi` with `curl`.'
            curl -fsSL https://pixi.sh/install.sh | sh
        elif can_exec wget; then
            print_debug '`wget` found, installing `pixi` with `wget`.'
            wget -qO- https://pixi.sh/install.sh | sh
        else
            print_error $'Can\'t install `pixi`: None of `wget` or `curl` found.'
            print_error 'Please install `wget` or `curl` via the system package manager.'
            return 1
        fi
    fi
}

# Performs the most important startup sanity checks and environment preparation.
function prepare_runtime_environment {
    # Ensure that pixi is installed.
    get_or_update_pixi

    # Get the right platform
    export OT_PLATFORM="$(get_platform)"

    # Install the environment
    install_env
}
