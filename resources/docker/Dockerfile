#To build, run
#    docker build -t <image-name> . -f resources/docker/Dockerfile
#    docker tag <image-name> <dockerhub-username>/<repository-name>:<tag>
#    docker push <dockerhub-username>/<repository-name>:<tag>

FROM ubuntu:latest

ARG OT_PLATFORM

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

ENV OT_PLATFORM="${OT_PLATFORM}"
ENV OT_SCRIPT_DEBUG="true"
ENV PATH="/root/.pixi/bin:${PATH}"
ENV TZ="Etc/UTC"
ENV OT_SCRIPT_DEBUG="true"

RUN --mount=type=cache,target=/root/.cache/rattler/cache <<EOF
pixi install --locked -v -e ${OT_PLATFORM} --skip OneTrainer
EOF

COPY . /OneTrainer

ENTRYPOINT [ "/usr/bin/bash", "/OneTrainer/docker.init.sh" ]
