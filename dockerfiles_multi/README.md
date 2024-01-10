# Dockerfiles

This directory contains the Dockerfiles used to build the NUWEST Docker images. 

## Building the Images

From the root of the repository, run the following command to build all images.
This will also update all images on Dockerhub to the latest version. 
This assumes you are logged in as Will, please ping him if you need to update the images :)


```bash
chmod +x dockerfiles/build.sh
./dockerfiles/update.sh
```

