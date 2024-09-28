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
# NOTE: The "OT_PYTHON_VENV" is always created relative to "SCRIPT_DIR" unless
# an absolute ("/home/foo/venv") or relative-traversal ("../venv") path is given.
# NOTE: The Conda detection prioritizes the user-provided value, otherwise the
# value of "$CONDA_EXE" (the env variable set by Conda's shell startup script),
# or lastly the "conda" binary (from PATH) as final fallback. We MUST use this
# order, otherwise we will fail to detect Conda if its startup script has executed,
# since their script shadows "conda" as a shell-function instead of a binary!
export OT_CONDA_CMD="${OT_CONDA_CMD:-${CONDA_EXE:-conda}}"
export OT_CONDA_ENV="${OT_CONDA_ENV:-onetrainer}"
export OT_PYTHON_CMD="${OT_PYTHON_CMD:-python}"
export OT_PYTHON_VENV="${OT_PYTHON_VENV:-.venv}"
export OT_PREFER_VENV="${OT_PREFER_VENV:-false}"
export OT_CUDA_LOWMEM_MODE="${OT_CUDA_LOWMEM_MODE:-false}"
export OT_PLATFORM_REQUIREMENTS="${OT_PLATFORM_REQUIREMENTS:-detect}"
export OT_SCRIPT_DEBUG="${OT_SCRIPT_DEBUG:-false}"

# Internal environment variables.
# NOTE: Version check supports "3", "3.1" and "3.1.5" specifier formats.
export OT_PYTHON_VERSION_MINIMUM="3"
export OT_PYTHON_VERSION_TOO_HIGH="3.11"
export OT_CONDA_USE_PYTHON_VERSION="3.10"
export OT_MUST_INSTALL_REQUIREMENTS="false"

# Force PyTorch to use fallbacks on Mac systems.
if [[ "$(uname)" == "Darwin" ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK="1"
fi

# Change PyTorch memory allocation to reduce CUDA out-of-memory situations.
if [[ "${OT_CUDA_LOWMEM_MODE}" == "true" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"
fi

# Utility functions.
function print {
    printf "[OneTrainer] %b\n" "$*"
}

function print_error {
    printf "Error: %b\n" "$*" >&2
}

function print_debug {
    if [[ "${OT_SCRIPT_DEBUG}" == "true" ]]; then
        print "$*"
    fi
}

function regex_escape {
    sed 's/[][\.|$(){}?+*^]/\\&/g' <<<"$*"
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

# Python command wrappers.
function run_python {
    print "+ ${OT_PYTHON_CMD} $*"
    "${OT_PYTHON_CMD}" "$@"
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
    print "+ ${OT_CONDA_CMD} $*"
    "${OT_CONDA_CMD}" "$@"
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
    # NOTE: We perform a strict, case-sensitive check for the exact env name.
    run_conda info --envs | grep -q -- "^$(regex_escape "${OT_CONDA_ENV}")\b"
}

function create_conda_env {
    print "Creating Conda environment with name \"${OT_CONDA_ENV}\"..."
    run_conda create -y -n "${OT_CONDA_ENV}" "python==${OT_CONDA_USE_PYTHON_VERSION}"
    export OT_MUST_INSTALL_REQUIREMENTS="true"
}

function ensure_conda_env_exists {
    if ! has_conda_env; then
        create_conda_env
    fi
}

function run_in_conda_env {
    # NOTE: The "--no-capture-output" flag is necessary to print live to stdout/stderr.
    run_conda run -n "${OT_CONDA_ENV}" --no-capture-output "$@"
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
        print "Using Conda environment with name \"${OT_CONDA_ENV}\"..."
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
        if can_exec nvidia-smi || can_exec nvcc; then
            # NOTE: NVIDIA drivers don't contain "nvcc". That's a CUDA dev-tool.
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
}

function install_requirements_in_active_env_if_necessary {
    if [[ "${OT_MUST_INSTALL_REQUIREMENTS}" != "false" ]]; then
        install_requirements_in_active_env
    fi
}

# Educates the user about the correct methods for installing Python or Conda.
function show_runtime_solutions {
    if should_use_conda; then
        # NOTE: We tell the user what to do, since automated removal is risky.
        print "Solution: Switch your Conda environment to the required Python version by deleting your old environment, and then run OneTrainer again.\n\nTo delete the outdated Conda environment, execute the following command:\n\"${OT_CONDA_CMD}\" remove -y -n \"${OT_CONDA_ENV}\" --all"
    else
        print "Solutions: Either install the required Python version via pyenv (https://github.com/pyenv/pyenv) and set the project directory's Python version with \"pyenv install <version>\" followed by \"pyenv local <version>\", or install Miniconda if you prefer that we automatically manage everything for you (https://docs.anaconda.com/miniconda/). Remember to manually delete any previous Venv or Conda environment which was created with a different Python version. Read \"LAUNCH-SCRIPTS.md\" for more detailed instructions."
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
    if [[ "$1" == "upgrade" ]]; then
        install_requirements_in_active_env
    else
        install_requirements_in_active_env_if_necessary
    fi
}
