#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <python_file> [--use-gpu]"
    exit 1
fi
if [ "$1" == "-h" ]; then
    echo "Runs the python script in the NUWEST container."
    echo "Usage: $0 <python_file> [--use-gpu]"
    exit 1
fi

if [ "$1" == "--help" ]; then
    echo "Runs the python script in the NUWEST container."
    echo "Usage: $0 <python_file> [--use-gpu]"
    exit 1
fi

if [ ! -f $1 ]; then
    echo "File $1 not found!"
    exit 1
fi

#Extract the python filename without path or extension
filename=$(basename -- "$1")
name="${filename%.*}"

echo "Running $filename..."
mkdir -p reports

if [ "$2" == "--use-gpu" ]; then
    echo "Running with GPU support..."
    apptainer run --cleanenv --nv utaustin_nuwest.sif python $1
else
    echo "Running without GPU support..."
    apptainer run --cleanenv utaustin_nuwest.sif python $1
fi