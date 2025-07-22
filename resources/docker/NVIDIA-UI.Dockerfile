# OneTrainer NVIDIA-UI Dockerfile (July 2025)
# Optimized for CUDA 12.x and Python 3.12 with GUI support

FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    OT_PLATFORM_REQUIREMENTS=requirements-cuda.txt \
    OT_LAZY_UPDATES=true \
    PIP_NO_CACHE_DIR=1

# System dependencies including full tkinter support
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    python3-tk \
    tk-dev \
    tcl-dev \
    libgtk-3-0 \
    libnotify4 \
    libnss3 \
    libxss1 \
    libxtst6 \
    xdg-utils \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3.12-tk \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && python3.12 -m pip install setuptools \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Clone OneTrainer
WORKDIR /
RUN git clone https://github.com/Nerogar/OneTrainer.git
WORKDIR /OneTrainer

# Install requirements
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip install -r requirements-cuda.txt \
    && pip install -r requirements-global.txt \
    && rm -rf ~/.cache/pip

# Set up entry point
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Default command to run the UI
CMD ["python", "scripts/train_ui.py"]
