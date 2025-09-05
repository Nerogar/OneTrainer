#!/bin/sh

set -e


# Allow users to provide a UID and GID to match their own, so files created inside the container retain the same numeric
# owner when mounted on the host.
if [ -n "$UID" ] && [ -n "$GID" ]; then
    echo "[entrypoint] Setting user UID and GID..."
    usermod  -u "$UID" onetrainer > /dev/null
    groupmod -g "$GID" onetrainer
else
    echo "[entrypoint] Missing UID or GID environment variables; keeping default values."
fi

# Fix file ownership
echo "[entrypoint] Changing /data ownership..."
chown -R onetrainer:onetrainer /data
chmod 777 .

# Add user to the /dev/nvidia* groups to ensure CUDA access. Normally, these devices belong to a single "video" group,
# but to be safe, we add the user to each device's group individually.
echo "[entrypoint] Adding user to GPU device groups..."
for dev in /dev/nvidia*; do
    # There's no universal standard for group IDs across Linux systems, so this may add 'onetrainer' to unusual groups.
    # For example, the 'video' group on some systems uses GID 27, which maps to 'sudo' in the python:3.12 image. This
    # should not cause serious issues.
    group=$(ls -ld "$dev" | awk '{print $4}')
    usermod -aG "$group" onetrainer
done

# Run the specified command as the onetrainer user
echo "[entrypoint] Running command..."
exec su onetrainer -c "$*"
