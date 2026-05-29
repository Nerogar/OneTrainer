#!/bin/bash
set -exo pipefail

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

# Login to HF
if [[ -n "${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}" ]]; then
  pixi run hf auth login --token "${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}" --add-to-git-credential 2>&1
else
  echo "HF_TOKEN or HUGGING_FACE_HUB_TOKEN not set; skipping login"
fi

# Login to WanDB
if [[ -n "${WANDB_API_KEY:-$WANDB_TOKEN}" ]]; then
  pixi run wandb login "${WANDB_API_KEY:-$WANDB_TOKEN}" 2>&1
else
  echo "WANDB_API_KEY or WANDB_TOKEN not set; skipping login"
fi

mkdir -p /workspace
ln -s /OneTrainer /workspace/OneTrainer

# Keep the container running
sleep infinity
