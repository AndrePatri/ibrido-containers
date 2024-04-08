FROM crizzard/lr-gym:2204-cudagl-basic
# FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	ca-certificates python3 python3-venv python-is-python3 \
        git wget sudo vim iputils-ping ssh rsync \
	dirmngr gnupg2 \
	build-essential \
	byobu \
	apt-utils \
	software-properties-common \
	curl \
        ffmpeg && rm -rf /var/lib/apt/lists/*
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata && rm -rf /var/lib/apt/lists/*

    # python-rosdep \
    # python-rosinstall \
    # python-vcstools \
# RUN ln -sv /usr/bin/python3 /usr/bin/python


# ----------------------------------------------------------------------
# System setup
# ----------------------------------------------------------------------

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
USER root
ENV PATH $PATH:/root/.local/bin

#Byobu Fix for launching BASH instead of SH
RUN mkdir -p /root/.byobu/
RUN echo 'set -g default-shell /bin/bash' >>/root/.byobu/.tmux.conf
RUN echo 'set -g default-command /bin/bash' >>/root/.byobu/.tmux.conf

RUN echo 'export PS1="\[\033[38;5;34m\]\u@\h\[$(tput sgr0)\]\[\033[38;5;76m\]:\[$(tput sgr0)\]\[\033[38;5;172m\]\w\[$(tput sgr0)\]\[\033[38;5;158m\]\\$\[$(tput sgr0)\] \[$(tput sgr0)\]"' >> /root/.bashrc

