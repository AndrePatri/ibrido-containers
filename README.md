# Docker Images and utilities for LRHControl

## docker 

Setting up docker in rootless mode: [how-to-rootless-docker](https://docs.docker.com/engine/security/rootless/).

Basically:
```
/usr/bin/dockerd-rootless-setuptool.sh install
docker context use rootless
```
```
systemctl --user start docker
```
or
```
systemctl --user restart docker
```

Before being able to run the `build_docker.sh` script, you need follow the instructions at [Omniverse's Isaac sim image](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html) and at [isaac-sim container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim), to get access to IsaacSim's container image.

To test
```
docker run -it --rm --gpus all ubuntu nvidia-smi
```

To build the image run 
```
build_docker.sh
```
This will create an image with isaac sim, micromamba and ros2. Utility scripts are installed in the `/root` folder. 
After having built the image, launch a persistent container with `launch_persistent_container.sh` and run, in this order `create_mamba_env.sh`, then `mamba activate LRHControlMambaEnv`, `source /opt/ros/humble/setup.bash` and, only after that, `setup_ws.sh`, which will build and setup the workspace.

You can now spawn the LRHControl by launching `launch_byobu_ws.sh`

## singularity