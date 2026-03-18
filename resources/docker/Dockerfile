#To build, run
#    docker build -t <image-name> . -f resources/docker/Dockerfile
#    docker tag <image-name> <dockerhub-username>/<repository-name>:<tag>
#    docker push <dockerhub-username>/<repository-name>:<tag>

FROM --platform=linux/amd64 ubuntu:24.04

ENV OT_PLATFORM=cuda
ENV OT_SCRIPT_DEBUG=true
ENV PATH=/root/.pixi/bin:${PATH}
ENV TZ=Etc/UTC

WORKDIR /
RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    ssh
ADD . /OneTrainer

RUN cd OneTrainer \
    && export OT_PLATFORM=cuda \
    && export OT_SCRIPT_DEBUG=true \
    && ./install.sh \
    && rm -rf /root/.cache \
    && mkdir -p /workspace \
    && ln -s /OneTrainer /workspace/OneTrainer

ENTRYPOINT [ "/usr/bin/bash /docker.init.sh" ]
