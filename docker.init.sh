#!/bin/bash
set -eo pipefail

setup_env() {
    # Export useful ENV variables, including all Runpod specific vars, to /etc/rp_environment
    # This file can then later be sourced in a login shell
    echo "Exporting environment variables..."
    printenv |
      grep -E '^RUNPOD_|^PATH=|^HF_HOME=|^HF_TOKEN=|^HUGGING_FACE_HUB_TOKEN=|^WANDB_API_KEY=|^WANDB_TOKEN=' |
      sed 's/^\(.*\)=\(.*\)$/export \1="\2"/' >> /etc/rp_environment

    # Add it to Bash login script only if it doesn't already exist
    grep -qxF 'source /etc/rp_environment' ~/.bashrc || echo 'source /etc/rp_environment' >> ~/.bashrc
    echo "cd /workspace/OneTrainer" >> ~/.bashrc

    source /etc/rp_environment
}

setup_ssh() {
    # Vast.ai uses $SSH_PUBLIC_KEY
    if [[ $SSH_PUBLIC_KEY ]]; then
      echo "INFO: Found SSH_PUBLIC_KEY, using it as PUBLIC_KEY"
      PUBLIC_KEY="${SSH_PUBLIC_KEY}"
    fi

    # Runpod uses $PUBLIC_KEY
    if [[ $PUBLIC_KEY ]]; then
      echo "INFO: Setting up SSH, adding PUBLIC_KEY to authorized_keys"
      mkdir -p ~/.ssh
      echo "${PUBLIC_KEY}" >> ~/.ssh/authorized_keys
      chmod 600 ~/.ssh/authorized_keys
      chmod 700 ~/.ssh
    fi

    # disable SSH password login - use key instead!
    sed -i -E 's/#?PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

    # Start SSH server
    service ssh start 2>&1
}

setup_auth() {
    # Login to HF
    if [[ -n "${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}" ]]; then
      pixi run --locked -e ${OT_PLATFORM} hf auth login --token "${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}" --add-to-git-credential 2>&1
    else
      echo "HF_TOKEN or HUGGING_FACE_HUB_TOKEN not set; skipping login"
    fi

    # Login to WanDB
    if [[ -n "${WANDB_API_KEY:-$WANDB_TOKEN}" ]]; then
      pixi run --locked -e ${OT_PLATFORM} wandb login "${WANDB_API_KEY:-$WANDB_TOKEN}" 2>&1
    else
      echo "WANDB_API_KEY or WANDB_TOKEN not set; skipping login"
    fi
}

setup_jupyterlab() {
    if [[ ! -n "${JUPYTER_PASSWORD}" ]]; then
        echo "INFO: Jupyter Lab not configured (JUPYTER_PASSWORD not set)"
        JUPYTER_PASSWORD=$(openssl rand -hex 32)
        echo "INFO: Jupyter Lab password set to: ${JUPYTER_PASSWORD}"
    fi
    echo "Starting Jupyter Lab..."
    mkdir -p /workspace &&
    cd / &&
    nohup jupyter-lab \
        --allow-root --no-browser --port=8888 --ip=* \
        --FileContentsManager.delete_to_trash=False \
        --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' \
        --IdentityProvider.token="${JUPYTER_PASSWORD}" \
        --ServerApp.allow_origin=* \
        --ServerApp.preferred_dir=/workspace \
        --ContentsManager.allow_hidden=True \
        --ServerApp.root_dir=/ \
        &> /jupyter.log &
    JUPYTER_PID=$!
    sleep 2
    if kill -0 "$JUPYTER_PID" 2>/dev/null; then
        echo "INFO: Jupyter Lab started (pid=$JUPYTER_PID)"
    else
        echo "ERROR: Jupyter Lab FAILED to start. /jupyter.log:" >&2
        cat /jupyter.log >&2
        return 1
    fi
}

setup_env
setup_ssh
setup_auth
setup_jupyterlab

mkdir -p /workspace
ln -s /OneTrainer /workspace/OneTrainer

# Keep the container running
sleep infinity
