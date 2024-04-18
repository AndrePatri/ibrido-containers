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
    libboost-all-dev\
	dirmngr gnupg2 \
	build-essential \
	byobu \
	wget \
	htop \
    nvtop \
	apt-utils \
	software-properties-common \
	nano \
	cmake \
	cmake-curses-gui \
	curl \
        ffmpeg && rm -rf /var/lib/apt/lists/*
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata && rm -rf /var/lib/apt/lists/*

# micromamba setup
ARG MAMBA_USER=mambauser
ARG MAMBA_USER_ID=57439
ARG MAMBA_USER_GID=57439
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"
COPY --from=mambaorg/micromamba:1.5.8 "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=mambaorg/micromamba:1.5.8 /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=mambaorg/micromamba:1.5.8 /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=mambaorg/micromamba:1.5.8 /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=mambaorg/micromamba:1.5.8 /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=mambaorg/micromamba:1.5.8 /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh
RUN echo "alias mamba='micromamba'" >> ~/.bashrc
RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh
SHELL ["/usr/local/bin/_dockerfile_shell.sh"]
COPY files/mamba_env.yml /root 

# RUN conda env create -y -f /root/mamba_env.yml # doesn't work because because during build docker cannot access GPU

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
    ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# isaaac
COPY files/nucleus.py /root/
RUN cp ~/nucleus.py /isaac-sim/exts/omni.isaac.core/omni/isaac/core/utils/nucleus.py

# Byobu Fix for launching BASH instead of SH
RUN mkdir -p /root/.byobu/
RUN echo 'set -g default-shell /bin/bash' >>/root/.byobu/.tmux.conf
RUN echo 'set -g default-command /bin/bash' >>/root/.byobu/.tmux.conf

# post-build setup
COPY files/setup.bash /root/setup.bash

COPY utils/setup_ws.sh /root/setup_ws.sh
RUN chmod +x /root/setup_ws.sh

COPY utils/create_mamba_env.sh /root/create_mamba_env.sh
RUN chmod +x /root/create_mamba_env.sh

COPY utils/post_build_setup.sh /root/post_build_setup.sh
RUN chmod +x /root/post_build_setup.sh
# ENTRYPOINT ["/root/post_build_setup.sh"]

WORKDIR /root
