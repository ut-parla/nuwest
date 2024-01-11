#!/bin/bash

# Usage instructions
usage() {
    echo "Runs the python script in the NUWEST container under the nsys profiler."
    echo "Usage: $0 <python_file> [python_script_args] [--use-gpu]"
}

# Check for minimum arguments
if [ $# -lt 1 ]; then
    usage
    exit 1
fi

# Help options
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    usage
    exit 0
fi

# Check if file exists and is a Python file
if [ ! -f "$1" ] || [[ $1 != *.py ]]; then
    echo "File $1 not found or is not a python file!"
    exit 1
fi

# Extract the python filename without path or extension for report naming
filename=$(basename -- "$1")
name="${filename%.*}"

# Check if the last argument is '--use-gpu'
use_gpu=false
if [ "${@: -1}" == "--use-gpu" ]; then
    use_gpu=true
    # Remove the '--use-gpu' argument
    set -- "${@:1:$#-1}"
fi

echo "Running $@..."
mkdir -p reports

if $use_gpu; then
    echo "Running with GPU support..."
    apptainer run --cleanenv --nv utaustin_nuwest.sif python $@
else
    echo "Running without GPU support..."
    apptainer run --cleanenv utaustin_nuwest.sif python $@
fi