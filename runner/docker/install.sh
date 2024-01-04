#!/bin/bash

DOCKER_IMAGE=wlruys/nuwest:cpu

# Read the image name from the command line
if [ $# -eq 1 ]; then
    DOCKER_IMAGE=$1
fi

# Provide a usage statement
if [ "$1" == "-h" ]; then
    echo "Pulls the latest version of the docker image and renames it."
    echo "Usage: $0 [docker_image]"
    exit 1
fi
if [ "$1" == "--help" ]; then
    echo "Pulls the latest version of the docker image and renames it."
    echo "Usage: $0 [docker_image]"
    exit 1
fi

# Pull the latest version of the docker image
docker pull $DOCKER_IMAGE

# Rename image to generic utaustin/nuwest
docker tag $DOCKER_IMAGE utaustin/nuwest
