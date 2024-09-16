#!/bin/bash

source ./linux-env-vars.sh

#additional arguments helpful if running into CUDA OOM error; default: false
use_alloc_args=false

if [ "$(uname)" = "Darwin" ]; then
	export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

if "$use_alloc_args"; then
	export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
fi

if ! [ -x "$(command -v ${python_cmd})" ]; then
	echo 'error: python not installed or found!'
elif [ -x "$(command -v ${python_cmd})" ]; then
	major=$(${python_cmd} -c 'import platform; major, minor, patch = platform.python_version_tuple(); print(major)')
	minor=$(${python_cmd} -c 'import platform; major, minor, patch = platform.python_version_tuple(); print(minor)')

	#check major version of python
	if [[ "$major" -eq "3" ]];
		then
			#check minor version of python
			if [[ "$minor" -le "10" ]];
				then
					if ! [ -x "$(command -v conda)" ]; then
						echo 'conda not found; python version correct; use native python'
						if ! [ -d $python_venv ]; then
							${python_cmd} -m venv $python_venv
						fi
						source $python_venv/bin/activate
						if [[ -z "$VIRTUAL_ENV" ]]; then
    							echo "warning: No VIRTUAL_ENV set. exiting."
						else
    							eval "$linux_cmd_python_venv"
						fi
					elif [ -x "$(command -v conda)" ]; then
						#check for venv
						if conda info --envs | grep -q ${conda_env}; 
							then
								bash --init-file <(echo ". \"$HOME/.bashrc\"; conda activate $conda_env; $linux_cmd_conda_venv")
							else 
								conda create -y -n $conda_env python==3.10;
								bash --init-file <(echo ". \"$HOME/.bashrc\"; conda activate $conda_env; $linux_cmd_conda_venv")
						fi
					fi
				else
					echo 'error: wrong python version installed:'$major'.'$minor
					echo 'OneTrainer requires the use of python 3.10, please refer to the anaconda project to setup a virtual environment with that version. https://anaconda.org/anaconda/python'
					exit 1
			fi
		else
			echo 'error: wrong python version installed:'$major'.'$minor
			echo 'OneTrainer requires the use of python 3.10, either install python3 on your system or refer to the anaconda project to setup a virtual environment with that version. https://anaconda.org/anaconda/python'
			exit 1
	fi
fi