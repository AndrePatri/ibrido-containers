## Container utilities for [IBRIDO](https://github.com/AndrePatri/IBRIDO) rapid deployment

### Singularity (now Apptainer)
Detailed instructions [here](https://apptainer.org/docs/admin/main/installation.html).
On Ubuntu, for the non setuid installation:
`sudo add-apt-repository -y ppa:apptainer/ppa`
`sudo apt update`
`sudo apt install -y apptainer`
On some systems, you may need a setuid installation. If you encounter errors like "permission denied" related to groups when trying to execute the container, then run the following:
`sudo add-apt-repository -y ppa:apptainer/ppa`
`sudo apt update`
`sudo apt install -y apptainer-suid'

### Docker (deprecated)

To set up docker in rootless mode see the [how-to-rootless-docker](https://docs.docker.com/engine/security/rootless/) guide or run the `isaac_ubuntu22/docker/utils/docker_rootless_installation.sh` script.

You can setup you workspace by running `isaac_ubuntu22/docker/create_ws.sh`. This script will create some directories used by the container + clone all the main packages of the echosystem. Please note that not all packages are currently publicly available (specifically the one hosting the employed RHC controller, which is owned by the [HHCM](https://hhcm.iit.it/) research line at IIT), so this step will fail if you don't have access to it. Basically, you need to be part of the [ADVR Humanoids](https://github.com/ADVRHumanoids) team.

Before being able to run the `isaac_ubuntu22/docker/build_docker.sh` script (which builds the base image used by LRHControl), you need follow the instructions at [Omniverse's Isaac sim image](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html) and at [isaac-sim container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim), to get access to IsaacSim's container image. 

Running `isaac_ubuntu22/docker/build_docker.sh` will build the base image, which is basically an image with [micromamba](https://github.com/mamba-org/micromamba-releases), [IsaacSim](https://developer.nvidia.com/isaac/sim) and [ros2-humble-base](https://docs.ros.org/en/humble/index.html).

After the build process has completed, you can run the `isaac_ubuntu22/docker/launch_persistent_container.sh` script and then, from the just opened bash shell, the `post_build_setup.sh` script. This will create the necessary mamba environment (cannot be done at build time due to [this issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/221)) and build/install the echosystem packages in the created `ibrido` micromamba environment. You will additionally be asked to login to [wandb](https://wandb.ai) (used for remote debugging).

You can now spawn the ibrido_ws [Byobu](https://www.byobu.org/) workspace by launching `launch_byobu_ws.sh`. From the workspace you can rapidly run all the main components of the echosystem. During a "minimal" training, you would just need to run the 3 already configured commands for the simulation environment, the control cluster and, finally, the training environment. Please note that the first time it may take a while due to IsaacSim's ray tracking shaders compilation and the RHC cluster having to codegenerate some symbolic function.

### Acknowledgements
Thanks to [c-rizz](https://github.com/c-rizz) for the technical support in setting up the container.
