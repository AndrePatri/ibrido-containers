FROM crizzard/lr-gym:2204-cudagl-basic

# preventing prompts that require user interaction during the package installation process.
ENV DEBIAN_FRONTEND noninteractive 

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
USER root
ENV PATH $PATH:/root/.local/bin

RUN apt-get update && apt-get install -y \
	ca-certificates python3 python3-venv python-is-python3 \
        git wget sudo vim iputils-ping ssh rsync \
	dirmngr gnupg2 \
	build-essential \
	byobu \
	wget \
	htop \
	apt-utils \
	software-properties-common \
	nano \
	cmake \
	cmake-curses-gui \
	curl \
        ffmpeg && rm -rf /var/lib/apt/lists/*
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata && rm -rf /var/lib/apt/lists/*

# mamba setup
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
RUN echo "alias mamba='micromamba'" >> ~/.bashrc
ENV PATH=$CONDA_DIR/bin:$PATH
COPY mamba_env.yml /root
RUN conda env create -y -f /root/mamba_env.yml 