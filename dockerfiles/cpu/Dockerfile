FROM nvidia/cuda:11.3.1-devel-ubuntu20.04 as build

RUN apt-get update \
    && apt-get install -y apt-utils \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get install -y git \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives /var/lib/apt/lists

ENV HOME_DIR /home/demo
RUN useradd --create-home --shell /bin/bash demo
USER demo
WORKDIR $HOME_DIR

# Install miniconda.
ENV MAMBA_DIR $HOME_DIR/mambaforge
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O Miniforge.sh && chmod +x Miniforge.sh && ./Miniforge.sh -b -p $HOME_DIR/mambaforge && rm Miniforge.sh
ENV PATH=$MAMBA_DIR/bin:$PATH
RUN mamba install -y python=3.11 && mamba init && echo "mamba activate base" >> ~/.bashrc

SHELL ["mamba", "run", "-n", "base", "/bin/bash", "-c"]
RUN python -m pip install -U pip setuptools && \
    python -m pip install numpy scikit-build scikit-build-core cython pytest pyyaml psutil numba jupyter ipython jupyterlab notebook && \
    mamba install pybind11>=2.11.1 patchelf>=0.17.2 cmake>=3.28

RUN git clone https://github.com/kokkos/pykokkos-base
RUN cd pykokkos-base && python setup.py install -- -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF -DENABLE_VIEW_RANKS=3 -DENABLE_CUDA=OFF -DENABLE_THREADS=OFF -DENABLE_OPENMP=ON -DCMAKE_CXX_STANDARD=17 && cd ..

ADD https://api.github.com/repos/kokkos/pykokkos/git/refs/heads/main pykokkos_version.json
RUN git clone https://github.com/kokkos/pykokkos
RUN cd pykokkos && python -m pip install . && cd ..

ADD https://api.github.com/repos/ut-parla/parla-experimental/git/refs/heads/main parla_version.json
RUN git clone https://github.com/ut-parla/parla-experimental.git
WORKDIR parla-experimental
RUN python -m pip install . --verbose
WORKDIR $HOME_DIR

FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

RUN export DEBIAN_FRONTEND=noninteractive \
    && export TX="America/New_York" \
    && apt-get update \
    && apt-get install -y apt-utils \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    dbus \
    fontconfig \
    gnupg \
    libasound2 \
    libfreetype6 \
    libglib2.0-0 \
    libnss3 \
    libsqlite3-0 \
    libx11-xcb1 \
    libxcb-glx0 \
    libxcb-xkb1 \
    libxcomposite1 \
    libxcb-xinput0 \
    libxcb-keysyms1 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libopengl0 \
    libegl1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxml2 \
    libxrandr2 \
    libxrender1 \
    libxtst6 \
    libgl1-mesa-glx \
    libxkbfile-dev \
    openssh-client \
    xcb \
    xkb-data \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives /var/lib/apt/lists

RUN sh -c 'echo kernel.perf_event_paranoid=2 > /etc/sysctl.d/local.conf'

RUN cd /tmp && \
    wget -q https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2023_4_1_97/nsight-systems-2023.4.1_2023.4.1.97-1_amd64.deb && \
    apt-get install -y ./nsight-systems-2023.4.1_2023.4.1.97-1_amd64.deb && \
    rm -rf /tmp/*

ENV HOME_DIR /home/demo
RUN useradd --create-home --shell /bin/bash demo
USER demo
WORKDIR $HOME_DIR

ENV MAMBA_DIR $HOME_DIR/mambaforge
COPY --from=build $MAMBA_DIR $MAMBA_DIR
ENV PATH=$MAMBA_DIR/bin:$PATH
RUN mamba init && echo "mamba activate base" >> ~/.bashrc
SHELL ["mamba", "run", "-n", "base", "/bin/bash", "-c"]

EXPOSE 8888