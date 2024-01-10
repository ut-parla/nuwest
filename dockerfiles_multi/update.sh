#!/bin/bash

# Update all Docker images
# Run all internal build.sh scripts for all subdirectories

# Assumes this script is called from the root directory
root_path=$(pwd)

for subdir in "dockerfiles_multi"/*; do
    if [ -d "$subdir" ]; then
        script_path="$subdir/build.sh"
        cd "$subdir"
        echo "Executing $script_path"
        chmod +x build.sh
        "./build.sh" &
        cd "$root_path"

    fi
done

# Wait for all scripts to finish