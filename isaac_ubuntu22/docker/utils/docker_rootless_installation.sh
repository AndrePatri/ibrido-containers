#!/bin/bash

# Install docker:
sudo apt update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" |   sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
# Check 
sudo docker run hello-world

# Enable Rootless mode
sudo apt-get install -y dbus-user-session
sudo systemctl disable --now docker.service docker.socket
sudo apt-get  install  -y uidmap
/usr/bin/dockerd-rootless-setuptool.sh install
systemctl --user start docker 
docker context use rootless
docker run hello-world

# Install nvidia packages
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg   && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
# Nvidia-container toolkit seems to have an issue with rootless mode.
# Something to do with cgroups, see https://github.com/NVIDIA/nvidia-docker/issues/1447
# To fix it I did disabled cgroups usage with the following (and then restarted docker):
sudo sed -i 's/#no-cgroups = false/no-cgroups = true/g' /etc/nvidia-container-runtime/config.toml
systemctl --user restart docker # remember to use --user!

# Check that the GPU can be seen:
docker run --rm --gpus all ubuntu nvidia-smi
