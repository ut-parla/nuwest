#!/bin/bash

# Provide a usage statement
if [ "$1" == "-h" ]; then
    echo "Launches a Jupyter notebook server from the NUWEST container."
    echo "Usage: $0 [--use-gpu]"
    exit 1
fi

# Check if the first argument is --use-gpu
if [ "$1" == "--use-gpu" ]; then
    echo "Running with GPU support..."
    docker run -p 8888:8888 --volume $(pwd):/app --workdir /app utaustin/nuwest jupyter notebook --ip="*" --NotebookApp.token='' -y --port=8888 --no-browser 

else
    echo "Running without GPU support..."
    docker run -p 8888:8888 --runtime=nvidia --gpus all --volume $(pwd):/app --workdir /app utaustin/nuwest jupyter notebook --ip="*" --NotebookApp.token='' -y --port=8888 --no-browser 
fi