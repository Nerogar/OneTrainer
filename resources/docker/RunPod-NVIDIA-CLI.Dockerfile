#To build, run
#    docker build -t <image-name> . -f RunPod-NVIDIA-CLI.Dockerfile
#    docker tag <image-name> <dockerhub-username>/<repository-name>:<tag>
#    docker push <dockerhub-username>/<repository-name>:<tag>

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
#the base image is barely used, pytorch is the wrong version. However, by using
#a base image that is popular on RunPod, the base image likely is already available
#in the image cache of a pod, and no download is necessary

WORKDIR /
RUN git clone https://github.com/Nerogar/OneTrainer
RUN cd OneTrainer \
 && export OT_PLATFORM_REQUIREMENTS=requirements-cuda.txt \
 && export OT_LAZY_UPDATES=true \
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
COPY RunPod-NVIDIA-CLI-start.sh.patch /start.sh.patch
RUN patch /start.sh < /start.sh.patch
