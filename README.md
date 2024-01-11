# UT Austin - NUWEST 

Welcome to the UT Austin NUWEST repository. 
Here we present installation, tutorials, and examples for two main core CS technologies from our PSAAPIII center:
- [PyKokkos](https://github.com/kokkos/pykokkos): A Python-to-Kokkos interface & JIT compiler 
- [Parla](https://github.com/ut-parla/parla-experimental): A Python thread-based task programming system for heterogeneous single node development

# Installation

We provide a Docker container at [`wlruys/nuwest`](https://hub.docker.com/repository/docker/wlruys/nuwest/general) for easy deployment as Kokkos may take over 30 minutes to build, but visitors are welcome to build from source using [the instructions below](#from-source) if their machine requires it.

## Running via the Container

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) or [Apptainer](https://apptainer.org/docs/user/latest)
- TODO: Test Podman on Lassen
- For GPU support: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
- 10GB Free HDD Space (for the container)

### Usage

If running on a TACC system, we use the [Apptainer](https://apptainer.org/docs/user/latest) container runtime. This can be loaded with the following command:

```bash
module load tacc-apptainer
```

#### Pull and rename the container: 

```bash
chmod +x runner/docker/*
./runner/docker/run.sh <container name>
```
Use the container that best matches your system:


| System | Container |
| --- | --- |
| CPU-only | `wlruys/nuwest:cpu` |
| CUDA 11.3 (SM70) | `wlruys/nuwest:volta-muli` |
| CUDA 11.3 (SM75) | `wlruys/nuwest:turing-multi` |
| CUDA 11.3 (SM80) | `wlruys/nuwest:ampere-multi` |

TODO: AMD Container for Tioga

#### Run a script

```bash
./runner/docker/run.sh lessons/parla/scripts/01_hello.py
```
This will run the script in the container and print the output to the terminal.

The `--use-gpu` flag is available to run on a GPU-enabled container.


#### Profile a script

```bash
./runner/docker/profile.sh lessons/parla/scripts/01_hello.py
```
The profile trace will be saved to `reports/<script name>.nsys-rep`.

The `--use-gpu` flag is available to run on a GPU-enabled container.

#### Launch a Jupyter Notebook Server

```
./runner/docker/notebook.sh
```
This will launch a Jupyter Notebook Server on port 8888 with password `NUWEST2024`. 

The `--use-gpu` flag is available to run on a GPU-enabled container.

## Connecting to a remote Jupyter Notebook Server

If you are running the container on a remote machine, you can connect to the Jupyter Notebook Server via SSH tunneling.

```
ssh -L <local_port>:localhost:8888 <username>@<remote machine>
```

Then, open a browser and navigate to `localhost:<local_port>` to access the Jupyter Notebook Server.

### Running on a SLURM Cluster

If you are running the container on a SLURM cluster, you must start the Jupyter Notebook Server on a compute node. You will need to request an allocation with 1 node and at least 1 GPU (if using a GPU-enabled container) and run the ./runner/docker/notebook.sh script on the compute node itself.

The connection will need to be forwarded back to your local machine via SSH tunneling. 
This can be done via a ProxyJump to the compute node through the login node.

Sample scripts are provided in `slurm/` to help automate this process.

While we have tested on TACC, note that the firewall settings on your cluster may prevent you from opening and forwarding the necessary ports. 


## Installing from Source

### Prerequisites
- `conda` or `mamba` (recommended) Python package manager
- `cmake` >= 3.28, `git`
- C++17 compatible compiler (e.g. `gcc` >= 7.3.0)
- CUDA 11.3 (optional, for GPU support)

### Mamba Installation (Linux)

Assumes a `sh` compatible shell in a Linux environment.

```bash
wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O Miniforge.sh
INSTALL_DIR=<location> bash Miniforge.sh -b -p $INSTALL_DIR/miniforge
mamba init
source $INSTALL_DIR/miniforge/etc/profile.d/conda.sh
mamba create -n nuwest_pykokkos python==3.11
conda activate nuwest_pykokkos 
```

In the following installations, we will assume that the `nuwest_pykokkos` environment is active. 
We will use the pip installer over mamba due to CuPy (and potentially numba) installations. 
pip will more easily build or stub against the system CUDA libraries without pulling in a potentially conflicting cuda-toolkit package. 

### PyKokkos Installation

```bash
# Install Python dependencies
pip install -U pip setuptools wheel
pip install numpy scikit-build pytest pyyaml psutil jupyter ipython jupyterlab notebook
mamba install pybind11>=2.11.1 cmake>=3.28 patchelf>=0.17.2
pip install cupy-cuda11x #(optional, for GPU support)

# Clone and install PyKokkos Base (compiles Kokkos)
git clone https://github.com/kokkos/pykokkos-base
cd pykokkos-base
python setup.py install -- \
-DENABLE_LAYOUTS=ON \ 
-DENABLE_OPENMP=ON \
-DCMAKE_CXX_STANDARD=17 \
-DENABLE_MEMORY_TRAITS=OFF \
-DENABLE_VIEW_RANKS=3 \
-DENABLE_THREADS=OFF \
-DENABLE_CUDA=ON \
-DKokkos_ARCH_TURING75=ON \
cd ..

# Clone and install PyKokkos Interface
git clone https://github.com/kokkos/pykokkos
cd pykokkos 
python -m pip install .
cd ..
```

The architecture flag `-DKokkos_ARCH_TURING75=ON` should be changed to match your GPU architecture.
See [Kokkos' documentation](https://kokkos.github.io/kokkos-core-wiki/keywords.html#architecture-keywords) for the list of architecture flags.

Note that building Kokkos may take a very long time as the `python setup.py install` command will build Kokkos from source and is currently single threaded.

To speed up the process you can manually build Kokkos with multiple threads using the following workaround. 

```bash
python setup.py build -- \
-DENABLE_LAYOUTS=ON \
-DENABLE_OPENMP=ON \
-DCMAKE_CXX_STANDARD=17 \
...
# all other flags

# Kill the PyKokkos-Base build process after it has started and reached (1%)
# Ctrl + C

cd  _scikit_build/<arch>/cmake_build
make -j <num threads>
cd ../../../
python setup.py install -- \
-DENABLE_LAYOUTS=ON \
-DENABLE_OPENMP=ON \
-DCMAKE_CXX_STANDARD=17 \
...
# all other flags



```


### Parla Installation

```bash
# Install Python dependencies
pip install numpy pyyaml psutil jupyter ipython jupyterlab notebook cython pytest scikit-build-core
pip install cupy-cuda11x #(optional, for GPU support)

# Clone and install Parla
git clone https://github.com/ut-parla/parla-experimental
cd parla-experimental
git submodule update --init --recursive
python -m pip install .
cd ..
```

### Nvidia Nsight Systems Installation
See [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) for more information.
Additional details and configuration is avaiable in [NVIDIA's documentation](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html#installing-nsight-systems).

The tutorial scripts assume that NSight Systems 2023.4 is available on the system path as `nsys-ui` and `nsys`. 
Older versions of NSight Systems will not support the `python-gil` trace option. 
This option can be removed from the tutorial scripts if necessary.
Mixing versions of `nsys` and `nsys-ui` may work but is not recommended.
