#!/bin/bash

# Let user specify python and venv directly, otherwise default to 'python', 'venv' and 'ot' respectively.

# python command to call
if [[ -z "${python_cmd}" ]]; then
   export  python_cmd="python"
fi
# python virtual environment name
if [[ -z "${python_venv}" ]]; then
    export python_venv=venv
fi
# conda virtual environment name
if [[ -z "${conda_venv}" ]]; then
    export conda_env=ot
fi

if [ -e /dev/kfd ]; then
	export PLATFORM_REQS=requirements-rocm.txt
elif [ -x "$(command -v nvcc)" ]; then
	export PLATFORM_REQS=requirements-cuda.txt
else
	export PLATFORM_REQS=requirements-default.txt
fi