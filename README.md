## Container utilities for [IBRIDO](https://github.com/AndrePatri/IBRIDO) rapid deployment

### 1) Setting up Singularity (now Apptainer)
First, you need to install Apptainer on your host system. Detailed installation instructions can be found [here](https://apptainer.org/docs/admin/main/installation.html).
On Ubuntu, for the non setuid installation:
`sudo add-apt-repository -y ppa:apptainer/ppa`
`sudo apt update`
`sudo apt install -y apptainer`
On some systems, you may need a setuid installation. If you encounter errors like "permission denied" related to groups when trying to execute the container, then run the following:
`sudo add-apt-repository -y ppa:apptainer/ppa`
`sudo apt update`
`sudo apt install -y apptainer-suid`

### 2) Build and setup container
- Clone this repo and then navigate to the container of choice. All the setup helper scripts follow the same API.
    As of now there are 3 different folders, each correspoding to a separated container:
    - `./ibrido_u22/singularity`: this container ships with Ubuntu 22 + IsaacSim + the IBRIDO framework + ROS2. This should be used to run trainings/evaluations exploiting the vectorized simulator's parallelization capabilities. 
    - `./ibrido_u20/singularity`: this is a lighter container with Ubuntu 20 + MuJoCo (CPU) + [XBot2](https://advrhumanoids.github.io/xbot2/v2.12.0/index.html) + the IBRIDO framework + ROS1. This is intended for sim-to-sim and sim-to-real evaluations through the XMjSimEnv and RtDeploymentEnv world interfaces, respectively, and does not currently support vectorized environments.
    - `./ibrido_u24/singularity`: same as `ibrido_u22`, but with Ubuntu 24 and the latest IsaacSim version (working, but integration with IBRIDO is WIP).

### Acknowledgements
Thanks to [c-rizz](https://github.com/c-rizz) for the technical support in setting up the containers.
