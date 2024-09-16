#!/bin/bash

source ./linux-env-vars.sh

export linux_cmd_python_venv='${python_cmd} scripts/train_ui.py'
export linux_cmd_conda_venv='${python_cmd} scripts/train_ui.py'

/bin/bash ./linux-python-env.sh