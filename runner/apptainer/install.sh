#!/bin/bash

DOCKER_IMAGE=wlruys/nuwest:cpu

# Read the image name from the command line
if [ $# -eq 1 ]; then
    DOCKER_IMAGE=$1
fi

# Pull the latest version of the docker image
apptainer pull utaustin_nuwest.sif docker://$DOCKER_IMAGE



