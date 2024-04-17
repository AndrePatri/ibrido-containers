# syntax=docker/dockerfile:1
FROM mambaorg/micromamba:1.5.8 as micromamba
FROM nvcr.io/nvidia/isaac-sim:2023.1.1

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
ARG MAMBA_USER=mambauser
ARG MAMBA_USER_ID=57439
ARG MAMBA_USER_GID=57439
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh
COPY mamba_env.yml /root
RUN echo "alias mamba='micromamba'" >> ~/.bashrc
RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh
SHELL ["/usr/local/bin/_dockerfile_shell.sh"]
RUN micromamba env create -y -f /root/mamba_env.yml 
##

# ros2-base
RUN echo 'Etc/UTC' > /etc/timezone && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*
RUN set -eux; \
       key='C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654'; \
       export GNUPGHOME="$(mktemp -d)"; \
       gpg --batch --keyserver keyserver.ubuntu.com --recv-keys "$key"; \
       mkdir -p /usr/share/keyrings; \
       gpg --batch --export "$key" > /usr/share/keyrings/ros2-latest-archive-keyring.gpg; \
       gpgconf --kill all; \
       rm -rf "$GNUPGHOME"
RUN echo "deb [ signed-by=/usr/share/keyrings/ros2-latest-archive-keyring.gpg ] http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2-latest.list
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    git \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*
ENV ROS_DISTRO humble
RUN rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO
RUN colcon mixin add default \
      https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml && \
    colcon mixin update && \
    colcon metadata add default \
      https://raw.githubusercontent.com/colcon/colcon-metadata-repository/master/index.yaml && \
    colcon metadata update
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base \
    && rm -rf /var/lib/apt/lists/*

# additional setup
COPY setup_docker.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/setup_docker.sh
RUN /usr/local/bin/setup_docker.sh


WORKDIR /root

COPY setup_ws.sh /root/
RUN chmod +x /root/setup_ws.sh