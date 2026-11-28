#!/usr/bin/env bash

set -eo pipefail

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
            local smi="nvidia-smi"
            if ! can_exec "${smi}"; then
                smi="/usr/lib/wsl/lib/nvidia-smi"
            fi
            local driver_cuda="$("${smi}" 2>/dev/null | grep -oE 'CUDA.* Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -n1)"
            if awk "BEGIN{exit !(${driver_cuda} >= 13.0)}"; then
                platform="cuda13"
            else
                print_warning "Detected CUDA version ${driver_cuda}. Updating to a driver that supports CUDA 13 is recommended."
                platform="cuda12"
            fi
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
    print_debug "Installing OneTrainer environment, this may take a while..."
    run_cmd pixi install --locked -e "${OT_PLATFORM}"
}

function run_in_env {
    run_cmd pixi run --locked -e "${OT_PLATFORM}" "$@"
}

function get_or_update_pixi {
    if ! can_exec pixi; then
        print_debug 'pixi not found, attempting to prepend to $PATH.'
        export PATH="${HOME:-/root}/.pixi/bin:$PATH"
    fi
    if can_exec pixi; then
        print_debug "pixi already available at $(which pixi)."
        if [[ "$1" == "upgrade" ]]; then
            print_debug 'pixi found, updating.'
            run_cmd pixi self-update --no-release-note
        fi
    else
        print_debug 'pixi not found, attempting to install.'
        ./install-pixi.sh
    fi
}

# Performs the most important startup sanity checks and environment preparation.
function prepare_runtime_environment {
    # Ensure that pixi is installed.
    get_or_update_pixi "$@"

    # Get the right platform
    export OT_PLATFORM="$(get_platform)"

    if [[ "$1" == "install" || "$1" == "upgrade" ]]; then
        # Install the environment
        install_env
    fi
}
