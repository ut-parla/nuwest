#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <python_file> [--use-gpu]"
    exit 1
fi
if [ "$1" == "-h" ]; then
    echo "Runs the python script in the NUWEST container under the nsys profiler."
    echo "Usage: $0 <python_file> [--use-gpu]"
    exit 1
fi

if [ "$1" == "--help" ]; then
    echo "Runs the python script in the NUWEST container under the nsys profiler."
    echo "Usage: $0 <python_file> [--use-gpu]"
    exit 1
fi

if [ ! -f $1 ]; then
    echo "File $1 not found!"
    exit 1
fi

#Check if file is a python file
if [[ $1 != *.py ]]; then
    echo "File $1 is not a python file!"
    exit 1
fi

#Extract the python filename without path or extension
filename=$(basename -- "$1")
name="${filename%.*}"

echo "Running $filename..."
mkdir -p reports

if [ "$2" == "--use-gpu" ]; then
    echo "Running with GPU support..."
    docker run --runtime=nvidia --gpus all --volume $(pwd):/app --workdir /app utaustin/nuwest nsys profile --force-overwrite true --trace nvtx,cuda,python-gil -o reports/${name}.nsys-rep python $1
else
    echo "Running without GPU support..."
    docker run --volume $(pwd):/app --workdir /app utaustin/nuwest nsys profile --force-overwrite true --trace nvtx,cuda,python-gil -o reports/${name}.nsys-rep python $1
fi