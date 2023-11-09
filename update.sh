#/bin/bash

#change the environment name conda for conda to use
conda_env=ot

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
						#TODO
						#git pull
						#python -m pip install -r requirements.txt --force-reinstall
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