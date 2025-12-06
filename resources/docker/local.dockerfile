# Dockerfile for the local OneTrainer GUI image. See compose.yaml for more information.
FROM python:3.12

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        libgl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create and move to the OneTrainer directory
WORKDIR /OneTrainer

# Install Python dependencies
COPY requirements.txt requirements-global.txt requirements-cuda.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install OneTrainer
COPY . .

# Create a persistent volume for runtime and user data. This setup can be used in three ways: the first is to do
# nothing, in which case Docker will automatically persist the volume; the second is to mount the /data directory
# entirely (e.g., -v ./data:/data); and the third (used by compose.yaml) is to mount each directory to its local
# counterpart.
RUN mkdir \
    # Regular data directories
        /data                        \
        /data/debug                  \
        /data/workspace              \
        /data/models                 \
        /data/training_concepts      \
        /data/training_samples       \
        /data/training_user_settings \
        /data/external               \
        /data/update.var           && \
    # training_presets directory, need to copy default presets
    mv training_presets /data      && \
    # Empty JSON files
    echo '{}' > /data/config.json  && \
    echo '{}' > /data/secrets.json && \
    # Create symbolic links from data directories to OneTrainer installation
    ln -s /data/* /OneTrainer

VOLUME [ "/data" ]

# Create runtime user, but don't switch to it yet. Switching should occur in the entrypoint script, after performing
# tasks requiring elevated privileges.
RUN useradd -m onetrainer

# Fix "dubious ownership" git error by marking the OneTrainer directory as safe
RUN git config --system safe.directory /OneTrainer

# Allow OneTrainer to create temporary *.write files at runtime.
RUN chmod 777 .

# Expose tensorboard
EXPOSE 6006

# Start trainer UI by default
RUN chmod +x resources/docker/entrypoint.sh
ENTRYPOINT ["resources/docker/entrypoint.sh"]
CMD ["python", "./scripts/train_ui.py"]
