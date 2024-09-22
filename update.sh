#!/bin/bash

source ./linux-env-vars.sh

export linux_cmd_python_venv='git pull && ${python_cmd} -m pip install -r requirements-global.txt -r $(<linux-platform.txt) --force-reinstall'
export linux_cmd_conda_venv='git pull; ${python_cmd} -m pip install -r requirements-global.txt -r $(<linux-platform.txt) --force-reinstall)'

/bin/bash ./linux-python-env.sh