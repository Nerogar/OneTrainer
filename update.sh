#!/bin/bash

source ./linux-env-vars.sh

stored_platform_reqs=$(<linux-platform)

export linux_cmd_python_venv = 'git pull && ${python_cmd} -m pip install -r requirements-global.txt -r $stored_platform_reqs --force-reinstall'
export linux_cmd_conda_venv = 'git pull; ${python_cmd} -m pip install -r requirements-global.txt -r $stored_platform_reqs --force-reinstall)'

/bin/bash ./linux-python-env.sh