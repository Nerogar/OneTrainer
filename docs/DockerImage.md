# Docker Image

A Dockerfile based on the `nvidia/cuda:11.8.0-devel-ubuntu22.04` is provided.

This image requires `nvidia-driver-525` and `nvidia-docker2` installed on the host.

## Building Image

Build using:

```
docker build -t myuser/onetrainer:latest -f Dockerfile .
```

## Running Image

This is an example

```
docker run \
  --gpus all \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -i \
  --tty \
  --shm-size=512m \
  myuser/onetrainer:latest
```
