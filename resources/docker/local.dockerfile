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

# Create user data volume
RUN mkdir /data && \
    mkdir /data/debug                 && ln -s /data/debug .                 && \
    mkdir /data/workspace             && ln -s /data/workspace .             && \
    mkdir /data/models                && ln -s /data/models .                && \
    mkdir /data/training_concepts     && ln -s /data/training_concepts .     && \
    mkdir /data/training_samples      && ln -s /data/training_samples .      && \
    mkdir /data/training_user_setting && ln -s /data/training_user_setting . && \
    mkdir /data/external              && ln -s /data/external .              && \
    mkdir /data/update.var            && ln -s /data/update.var .            && \
    echo '{}' > /data/config.json     && ln -s /data/config.json .           && \
    echo '{}' > /data/secrets.json    && ln -s /data/secrets.json .

VOLUME [ "/data" ]

# Create runtime user, but don't switch to it yet. Switching should occur in the entrypoint script, after performing
# tasks requiring elevated privileges.
RUN useradd -m onetrainer

# Install OneTrainer
COPY . .
# Fix "dubious ownership" git error by marking the OneTrainer directory as safe
RUN git config --system safe.directory /OneTrainer

# Move the default training presets to the persistant data volume, and overwrite the training preset directory with a
# link pointing back to that volume.
RUN mkdir /data/training_presets                 && \
    mv training_presets/* /data/training_presets && \
    rm -r training_presets
RUN ln -s /data/training_presets /OneTrainer/training_presets
# Allow OneTrainer to create temporary *.write files at runtime.
RUN chmod 777 .

# Expose tensorboard
EXPOSE 6006

# Start trainer UI by default
RUN chmod +x resources/docker/entrypoint.sh
ENTRYPOINT ["resources/docker/entrypoint.sh"]
CMD ["python", "./scripts/train_ui.py"]
