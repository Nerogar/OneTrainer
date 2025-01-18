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
# NOTE: The "OT_CONDA_ENV" and "OT_PYTHON_VENV" are always relative to "SCRIPT_DIR"
# unless an absolute ("/home/foo/venv") or relative-traversal ("../venv") path is given.
# NOTE: The Conda detection prioritizes the user-provided value, otherwise the
# value of "$CONDA_EXE" (the env variable set by Conda's shell startup script),
# or lastly the "conda" binary (from PATH) as final fallback. We MUST use this
# order, otherwise we will fail to detect Conda if its startup script has executed,
# since their script shadows "conda" as a shell-function instead of a binary!
export OT_CONDA_CMD="${OT_CONDA_CMD:-${CONDA_EXE:-conda}}"
export OT_CONDA_ENV="${OT_CONDA_ENV:-conda_env}"
export OT_PYTHON_CMD="${OT_PYTHON_CMD:-python}"
export OT_PYTHON_VENV="${OT_PYTHON_VENV:-venv}"
export OT_PREFER_VENV="${OT_PREFER_VENV:-false}"
export OT_LAZY_UPDATES="${OT_LAZY_UPDATES:-false}"
export OT_CUDA_LOWMEM_MODE="${OT_CUDA_LOWMEM_MODE:-false}"
export OT_PLATFORM_REQUIREMENTS="${OT_PLATFORM_REQUIREMENTS:-detect}"
export OT_SCRIPT_DEBUG="${OT_SCRIPT_DEBUG:-false}"

# Internal environment variables.
# NOTE: Version check supports "3", "3.1" and "3.1.5" specifier formats.
export OT_PYTHON_VERSION_MINIMUM="3.10"
export OT_PYTHON_VERSION_TOO_HIGH="3.13"
export OT_CONDA_USE_PYTHON_VERSION="3.10"
export OT_MUST_INSTALL_REQUIREMENTS="false"
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

# Python command wrappers.
function run_python {
    run_cmd "${OT_PYTHON_CMD}" "$@"
}

function run_pip {
    run_python -m pip "$@"
}

function run_venv {
    run_python -m venv "$@"
}

function has_python {
    can_exec "${OT_PYTHON_CMD}"
}

function has_python_venv {
    [[ -f "${OT_PYTHON_VENV}/bin/activate" ]]
}

function create_python_venv {
    print "Creating Python Venv environment in \"${OT_PYTHON_VENV}\"..."
    run_venv "${OT_PYTHON_VENV}"
    export OT_MUST_INSTALL_REQUIREMENTS="true"
}

function ensure_python_venv_exists {
    if ! has_python_venv; then
        create_python_venv
    fi
}

function activate_python_venv {
    # NOTE: This rewrites PATH to make all subsequent Python commands prefer
    # to use the venv's binaries instead. You should only execute this once!
    source "${OT_PYTHON_VENV}/bin/activate"

    # NOTE: Sanity check just to ensure that the activate-script was real.
    if [[ -z "${VIRTUAL_ENV}" ]]; then
        print_error "Something went wrong when activating the Python Venv in \"${OT_PYTHON_VENV}\"."
        exit 1
    fi

    # We must now force the Python binary name back to normal, since the venv's
    # own, internal Python binary is ALWAYS named "python".
    export OT_PYTHON_CMD="python"
}

# Conda command wrappers.
function run_conda {
    run_cmd "${OT_CONDA_CMD}" "$@"
}

__HAS_CONDA__CACHE=""
function has_conda {
    # We cache the result of this check to speed up further "has_conda" calls.
    if [[ -z "${__HAS_CONDA__CACHE}" ]]; then
        if can_exec "${OT_CONDA_CMD}"; then
            __HAS_CONDA__CACHE="true"
        else
            __HAS_CONDA__CACHE="false"
        fi
    fi

    [[ "${__HAS_CONDA__CACHE}" == "true" ]]
}

function has_conda_env {
    # Look for Conda's metadata to ensure it's a valid local Conda environment.
    [[ -d "${OT_CONDA_ENV}/conda-meta" ]]
}

function has_conda_global_env {
    if [[ -z "$1" ]]; then
        print_error "has_conda_global_env requires 1 argument."
        return 1
    fi

    # Checks for a globally installed (non-local) Conda environment by name.
    # NOTE: We perform a strict, case-sensitive check for the exact env name.
    run_conda info --envs | grep -q -- "^$(regex_escape "$1")\b"
}

function create_conda_env {
    print "Creating Conda environment in \"${OT_CONDA_ENV}\"..."

    # IMPORTANT: The ".*" suffix tells Conda to install the latest bugfix/patch
    # release of the desired Python version. For example, if we specify "3.12.*",
    # then it will pick the latest patch release, such as "3.12.5". It also works
    # correctly if we specify an EXACT patch release ourselves, such as "3.10.14.*",
    # or if we only specify a major version, such as "3.*" (gets the latest release).
    declare -a install_args=()
    install_args+=("python==${OT_CONDA_USE_PYTHON_VERSION}.*")

    # IMPORTANT: We MUST use "conda-forge" and EXPLICITLY switch to the version
    # of Tk that has libXft support on Linux, otherwise the GUI will have broken
    # fonts, inability to render Unicode, and no antialiasing! Doesn't affect Macs.
    # SEE: https://github.com/conda-forge/tk-feedstock/pull/40#issuecomment-2381409555
    # SEE: https://anaconda.org/conda-forge/tk/files (only Linux has "xft" variant).
    if [[ "${OT_HOST_OS}" == "Linux" ]]; then
        install_args+=("tk[build=xft_*]")
    fi

    # NOTE: We install with strict channel priority and an explicit channel list,
    # which ensures that package names which exist in "conda-forge" will never
    # fall back to the "defaults" channel if "conda-forge" lacks the required
    # version. Protects against mismatched packages built with different settings.
    run_conda create -y --prefix "${OT_CONDA_ENV}" --override-channels --strict-channel-priority --channel "conda-forge" "${install_args[@]}"
    export OT_MUST_INSTALL_REQUIREMENTS="true"

    # Show a warning if the user has the legacy "ot" environment on their system.
    if has_conda_global_env "ot"; then
        # NOTE: We tell the user what to do, since automated removal is risky.
        print_warning "The deprecated \"ot\" Conda environment has been detected on your system. It is occupying several gigabytes of disk space, and can be deleted manually to reclaim the storage space.\n\nTo delete the outdated Conda environment, execute the following command:\n\"${OT_CONDA_CMD}\" remove -y --name \"ot\" --all"
    fi
}

function ensure_conda_env_exists {
    if ! has_conda_env; then
        create_conda_env
    fi
}

function run_in_conda_env {
    # NOTE: The "--no-capture-output" flag is necessary to print live to stdout/stderr.
    run_conda run --prefix "${OT_CONDA_ENV}" --no-capture-output "$@"
}

function run_python_in_conda_env {
    # NOTE: Python is ALWAYS called "python" inside Conda's environment.
    run_in_conda_env python "$@"
}

function run_pip_in_conda_env {
    run_python_in_conda_env -m pip "$@"
}

# Checks if the user hasn't requested Venv instead, and if Conda exists.
function should_use_conda {
    # NOTE: This check is intentionally not cached, to allow changing preference
    # during runtime. Furthermore, "has_conda" uses caching for speed already.
    [[ "${OT_PREFER_VENV}" != "true" ]] && has_conda
}

# Helpers which automatically run Python and Pip in either Conda or Venv/Host,
# depending on what's available on the system or user-preference overrides.
function activate_chosen_env {
    if should_use_conda; then
        print "Using Conda environment in \"${OT_CONDA_ENV}\"..."
        ensure_conda_env_exists
    else
        print "Using Python Venv environment in \"${OT_PYTHON_VENV}\"..."
        ensure_python_venv_exists
        activate_python_venv
    fi
}

function run_python_in_active_env {
    if should_use_conda; then
        run_python_in_conda_env "$@"
    else
        run_python "$@"
    fi
}

function run_pip_in_active_env {
    if should_use_conda; then
        run_pip_in_conda_env "$@"
    else
        run_pip "$@"
    fi
}

# Determines which requirements.txt file we need to install.
function get_platform_requirements_path {
    # NOTE: The user can override our platform detection via the environment.
    local platform_reqs="${OT_PLATFORM_REQUIREMENTS}"
    if [[ "${platform_reqs}" == "detect" ]]; then
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
            platform_reqs="requirements-cuda.txt"
        elif [[ -e "/dev/kfd" ]]; then
            # AMD graphics.
            platform_reqs="requirements-rocm.txt"
        else
            # No GPU acceleration.
            platform_reqs="requirements-default.txt"
        fi
    fi

    if [[ -z "${platform_reqs}" ]] || [[ ! -f "${platform_reqs}" ]]; then
        print_error "Requirements file \"${platform_reqs}\" does not exist."
        return 1
    fi

    echo "${platform_reqs}"
}

# Installs the Global and Platform requirements into the active environment.
function install_requirements_in_active_env {
    # Ensure that we have the latest Python tools, and install the dependencies.
    # NOTE: The "eager" upgrade strategy is necessary for upgrading dependencies
    # when running in existing environments. It ensures that all libraries will
    # be upgraded to the same versions as a fresh reinstall of requirements.txt.
    print "Installing requirements in active environment..."
    run_pip_in_active_env install --upgrade --upgrade-strategy eager pip setuptools
    run_pip_in_active_env install --upgrade --upgrade-strategy eager -r requirements-global.txt -r "$(get_platform_requirements_path)"
    export OT_MUST_INSTALL_REQUIREMENTS="false"

    # Write update-check metadata to disk if user has requested "lazy updates",
    # otherwise delete any old, leftover metadata to avoid clutter.
    if [[ "${OT_LAZY_UPDATES}" == "true" ]]; then
        print_debug "Saving current update-check metadata to disk..."
        save_update_metadata
    elif [[ -f "${OT_UPDATE_METADATA_FILE}" ]]; then
        print_debug "Deleting outdated update-check metadata from disk..."
        rm -f "${OT_UPDATE_METADATA_FILE}"
    fi
}

function install_requirements_in_active_env_if_necessary {
    if [[ "${OT_MUST_INSTALL_REQUIREMENTS}" != "false" ]]; then
        install_requirements_in_active_env
    fi
}

# Educates the user about the correct methods for installing Python or Conda.
function show_runtime_solutions {
    if should_use_conda; then
        # Resolve the absolute path to ensure user doesn't delete anything else.
        local conda_env_path="${OT_CONDA_ENV}"
        if has_conda_env; then
            conda_env_path="$(absolute_path "${conda_env_path}")"
        fi

        # NOTE: We tell the user what to do, since automated removal is risky.
        print "Solution: Switch your Conda environment to the required Python version by deleting your old environment, and then run OneTrainer again.\n\nTo delete the outdated Conda environment, execute the following command:\n\"${OT_CONDA_CMD}\" remove -y --prefix \"${conda_env_path}\" --all"
    else
        print "Solution: Either install the required Python version via pyenv (https://github.com/pyenv/pyenv) and set the project directory's Python version with \"pyenv install <version>\" followed by \"pyenv local <version>\", or install Miniconda if you prefer that we automatically manage everything for you (https://docs.anaconda.com/miniconda/). Remember to manually delete any previous Venv or Conda environment which was created with a different Python version. Read \"LAUNCH-SCRIPTS.md\" for more detailed instructions."
    fi
}

# Ensures that Python or Conda exists on the host and can be executed.
function exit_if_no_runtime {
    # NOTE: If "should_use_conda" is true, we have a usable Conda.
    if ! should_use_conda && ! has_python; then
        print_error "Python command \"${OT_PYTHON_CMD}\" does not exist on your system."
        show_runtime_solutions
        exit 1
    fi
}

# Verifies that Python version is ">= minimum and < too high" in Conda/Venv/Host.
function exit_if_active_env_wrong_python_version {
    if ! run_python_in_active_env "scripts/util/version_check.py" "${OT_PYTHON_VERSION_MINIMUM}" "${OT_PYTHON_VERSION_TOO_HIGH}"; then
        show_runtime_solutions
        exit 1
    fi
}

# Performs the most important startup sanity checks and environment preparation.
function prepare_runtime_environment {
    # Ensure that the chosen Conda or Python runtime exists.
    exit_if_no_runtime

    # Create and activate the chosen environment.
    activate_chosen_env

    # Protect against outdated Python environments created with older versions.
    exit_if_active_env_wrong_python_version

    # If this is an upgrade, always ensure that we have the latest dependencies,
    # otherwise only install requirements if the environment was newly created.
    # NOTE: If "OT_LAZY_UPDATES" is "true", we will check the last update status
    # to determine if the source code has changed since our previous reqs update,
    # otherwise we'll fall back to the normal "only update if new env" mode.
    local force_update="false"
    if [[ "$1" == "upgrade" ]]; then
        if [[ "${OT_LAZY_UPDATES}" == "false" ]] || is_update_metadata_changed; then
            force_update="true"
        fi
    fi
    if [[ "${force_update}" == "true" ]]; then
        print_debug "Triggering a forced update of the environment's dependencies..."
        install_requirements_in_active_env
    else
        install_requirements_in_active_env_if_necessary
    fi
}
