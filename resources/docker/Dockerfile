#To build, run
#    docker build -t <image-name> . -f resources/docker/Dockerfile
#    docker tag <image-name> <dockerhub-username>/<repository-name>:<tag>
#    docker push <dockerhub-username>/<repository-name>:<tag>

FROM ubuntu:latest

ENV OT_PLATFORM=cuda
ENV OT_SCRIPT_DEBUG=true
ENV PATH=/root/.pixi/bin:${PATH}
ENV TZ=Etc/UTC
ENV OT_SCRIPT_DEBUG=true
ENV CONDA_OVERRIDE_CUDA=13.0

WORKDIR /
RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    ssh

WORKDIR /OneTrainer

COPY --link ./install-pixi.sh ./

RUN ./install-pixi.sh

COPY --link pyproject.toml pixi.lock ./

RUN --mount=type=cache,target=/root/.cache/rattler/cache <<EOF
pixi install --locked -v -e ${OT_PLATFORM} --skip OneTrainer
EOF

COPY . /OneTrainer

ENTRYPOINT [ "/usr/bin/bash", "/OneTrainer/docker.init.sh" ]
