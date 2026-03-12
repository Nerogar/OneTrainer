#To build, run
#    docker build -t <image-name> . -f Vast-NVIDIA-CLI.Dockerfile
#    docker tag <image-name> <dockerhub-username>/<repository-name>:<tag>
#    docker push <dockerhub-username>/<repository-name>:<tag>

FROM vastai/pytorch:cuda-12.8.1-auto

WORKDIR /
USER root
RUN git clone https://github.com/Nerogar/OneTrainer
RUN cd OneTrainer \
 && export OT_PLATFORM_REQUIREMENTS=requirements-cuda.txt \
 && export OT_LAZY_UPDATES=true \
 && export OT_PYTHON_CMD=/venv/main/bin/python \
 && ./install.sh \
 && pip cache purge \
 && rm -r ~/.cache/pip
RUN apt-get update --yes \
 && apt-get install --yes --no-install-recommends \
      joe \
	  less \
	  gh \
	  iputils-ping \
	  nano \
	  nethogs \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
RUN pip install nvitop \
 && pip cache purge \
 && rm -rf ~/.cache/pip
RUN mkdir /workspace && ln -snf /OneTrainer /workspace/OneTrainer
