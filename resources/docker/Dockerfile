#To build, run
#    docker build -t <image-name> . -f resources/docker/Dockerfile --build-arg OT_PLATFORM=[cpu|cuda12|cuda13|rocm] --no-cache-filter=final
#    docker tag <image-name> <dockerhub-username>/<repository-name>:<tag>
#    docker push <dockerhub-username>/<repository-name>:<tag>

FROM ubuntu:latest AS base

ARG OT_PLATFORM
ARG OT_REPO="https://github.com/Nerogar/OneTrainer.git"
ARG OT_BRANCH="master"

WORKDIR /
RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    ssh \
    jupyterlab \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /OneTrainer

COPY --link ./install-pixi.sh ./

RUN ./install-pixi.sh

COPY --link pyproject.toml pixi.lock ./

ENV OT_SCRIPT_DEBUG="true"
ENV PATH="/root/.pixi/bin:${PATH}"
ENV TZ="Etc/UTC"
ENV OT_SCRIPT_DEBUG="true"

RUN --mount=type=cache,target=/root/.cache/rattler/cache <<EOF
pixi install --locked -v -e ${OT_PLATFORM} --skip OneTrainer
EOF

FROM base AS final

RUN git init \
    && git remote add origin ${OT_REPO} \
    && git fetch origin ${OT_BRANCH} \
    && git reset --hard origin/${OT_BRANCH} \
    && git checkout ${OT_BRANCH}

ENTRYPOINT [ "/usr/bin/bash", "/OneTrainer/docker.init.sh" ]
