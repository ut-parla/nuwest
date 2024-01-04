#!/bin/bash

# Provide a usage statement
if [ "$1" == "-h" ]; then
    echo "Launches a Jupyter notebook server from the NUWEST container."
    echo "Usage: $0 [--use-gpu]"
    exit 1
fi

PASSWORD_HASH='sha1:b65489bbd2d0:7f0c0ba157c5e88132db52a6f5a726dbc9ce4d0a'
# Check if the first argument is --use-gpu
if [ "$1" == "--use-gpu" ]; then
    echo "Running with GPU support..."
    docker run -p 8888:8888 --runtime=nvidia --gpus all --volume $(pwd):/app --workdir /app utaustin/nuwest jupyter notebook --ip="*" --PasswordIdentityProvider.hashed_password=$PASSWORD_HASH -y --port=8888 --no-browser 
else
    echo "Running without GPU support..."
    docker run -p 8888:8888 --volume $(pwd):/app --workdir /app utaustin/nuwest jupyter notebook --ip="*" --PasswordIdentityProvider.hashed_password=$PASSWORD_HASH -y --port=8888 --no-browser 
fi
