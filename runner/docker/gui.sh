#!/bin/bash

# Native install:
nsys-ui 

# MACOS via Docker:
# open -a xquartz
# xhost +local
# docker run \
#     -e DISPLAY=host.docker.internal:0 \
#     -v $(pwd):/app \
#     -w /app \
#     utaustin/nuwest \
#     nsys-ui
# xhost -local

# LINUX via Docker:
# xhost +local
# docker run \
#     -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     -v $(pwd):/app \
#     -w /app \
#     utaustin/nuwest \
#     nsys-ui
# xhost -local


