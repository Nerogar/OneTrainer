#!/bin/bash

source ./linux-env-vars.sh

export linux_cmd_python_venv = '${python_cmd} -m pip install -r requirements-global.txt -r $PLATFORM_REQS'
export linux_cmd_conda_venv = '${python_cmd} -m pip install -r requirements-global.txt -r $PLATFORM_REQS'

/bin/bash ./linux-python-env.sh