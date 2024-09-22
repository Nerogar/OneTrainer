#!/bin/bash

source ./linux-env-vars.sh

ask() {
    local prompt default reply

    if [[ ${2:-} = 'Y' ]]; then
        prompt='Y/n'
        default='Y'
    elif [[ ${2:-} = 'N' ]]; then
        prompt='y/N'
        default='N'
    else
        prompt='y/n'
        default=''
    fi

    while true; do

        # Ask the question (not using "read -p" as it uses stderr not stdout)
        echo -n "$1 [$prompt] "

        # Read the answer (use /dev/tty in case stdin is redirected from somewhere else)
        read -r reply </dev/tty

        # Default?
        if [[ -z $reply ]]; then
            reply=$default
        fi

        # Check if the reply is valid
        case "$reply" in
            Y*|y*) return 0 ;;
            N*|n*) return 1 ;;
        esac

    done
}

#clear linux-platform (used to store the selected platform-requirements, so we don't have to confirm on every update)
> linux-platform.txt

echo ""
echo "Platform requirements diverge slightly depending on you GPU.Would you want to use the detected platform Requirements? ($PLATFORM_REQS)"
if ask "Use $PLATFORM_REQS? default: Yes" Y; then
    echo "Using $PLATFORM_REQS"
    echo $PLATFORM_REQS > linux-platform.txt
else
    echo "Using default"
    echo "requirements-default.txt" > linux-platform.txt
fi

export linux_cmd_python_venv='${python_cmd} -m pip install -r requirements-global.txt -r $(<linux-platform.txt)'
export linux_cmd_conda_venv='${python_cmd} -m pip install -r requirements-global.txt -r $(<linux-platform.txt)'

/bin/bash ./linux-python-env.sh