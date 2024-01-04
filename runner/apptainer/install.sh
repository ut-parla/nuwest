#!/bin/bash

DOCKER_IMAGE=wlruys/nuwest:cpu

# Pull the latest version of the docker image
apptainer pull utaustin_nuwest.sif docker://$DOCKER_IMAGE



