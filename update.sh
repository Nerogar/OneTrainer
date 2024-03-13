#!/bin/bash

#change the environment name for conda to use
conda_env=ot
#change the environment name for python to use (only needed if Anaconda3 or miniconda is not installed)
python_venv=venv

if ! [ -x "$(command -v python)" ]; then
	echo 'error: python not installed or found!'
	break
elif [ -x "$(command -v python)" ]; then
	major=$(python -c 'import platform; major, minor, patch = platform.python_version_tuple(); print(major)')
	minor=$(python -c 'import platform; major, minor, patch = platform.python_version_tuple(); print(minor)')

	#check major version of python
	if [[ "$major" -eq "3" ]];
		then
			#check minor version of python
			if [[ "$minor" -le "10" ]];
				then
					if ! [ -x "$(command -v conda)" ]; then
						echo 'conda not found; python version correct; use native python'
						git pull
						if ! [ -d $python_venv ]; then
							python -m venv $python_venv
						fi
						source $python_venv/bin/activate
						if [[ -z "$VIRTUAL_ENV" ]]; then
    							echo "warning: No VIRTUAL_ENV set. exiting."
						else
							REQS_INSTALLED=0
							if [ -e /dev/kfd ]; then
								python -m pip install -r requirements-global.txt -r requirements-rocm.txt --force-reinstall
								REQS_INSTALLED=1
							fi
							if [ -x "$(command -v nvidia-smi)" ]; then
								python -m pip install -r requirements-global.txt -r requirements-cuda.txt --force-reinstall
								REQS_INSTALLED=1
							fi
							if [ $REQS_INSTALLED -eq 0 ]; then
								python -m pip install -r requirements-global.txt -r requirements-default.txt --force-reinstall
							fi
						fi
					elif [ -x "$(command -v conda)" ]; then
						#check for venv
						if conda info --envs | grep -q ${conda_env}; 
							then
								bash --init-file <(echo ". \"$HOME/.bashrc\"; conda activate $conda_env; git pull; python -m pip install -r requirements.txt" --force-reinstall)
							else 
								echo 'run install.sh first'
						fi
					fi
				else
					echo 'error: wrong python version installed:'$major'.'$minor
					echo 'OneTrainer requires the use of python 3.10, please refer to the anaconda project to setup a virtual environment with that version. https://anaconda.org/anaconda/python'
					break
			fi
		else
			echo 'error: wrong python version installed:'$major'.'$minor
			echo 'OneTrainer requires the use of python 3.10, either install python3 on your system or refer to the anaconda project to setup a virtual environment with that version. https://anaconda.org/anaconda/python'
			break
	fi
fi