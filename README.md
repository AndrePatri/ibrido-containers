## Container utilities for [IBRIDO](https://github.com/AndrePatri/IBRIDO) rapid deployment

### 0) Why not Docker?
Even though Docker is a very popular choice, IBRIDO uses Apptainer. Why?

Apptainer is designed for HPC environments, which usually do not allow Docker usage at all. It emphasizes integration with the host system (VS Docker's isolation), offers stronger security through non-privileged container execution, and easier integration with batch schedulers and parallel computing workflows.

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
    As of now there are 3 different folders, each corresponding to a separated container:
    - `./ibrido_u22/singularity`: this container ships with Ubuntu 22 + IsaacSim + the IBRIDO framework + ROS2. This should be used to run trainings/evaluations exploiting the vectorized simulator's parallelization capabilities. 
    - `./ibrido_u20/singularity`: this is a lighter container with Ubuntu 20 + MuJoCo (CPU) + [XBot2](https://advrhumanoids.github.io/xbot2/v2.12.0/index.html) + the IBRIDO framework + ROS1. This is intended for sim-to-sim and sim-to-real evaluations through the XMjSimEnv and RtDeploymentEnv world interfaces, respectively, and does not currently support vectorized environments. 
    - `./ibrido_u24/singularity`: same as `ibrido_u22`, but with Ubuntu 24 and the latest IsaacSim version (working, but integration with IBRIDO is WIP).
- Navigate to `./ibrido_u*/singularity` and run `export IBRIDO_CONTAINERS_PREFIX=$PWD`
- Run `setup_container.sh -i -b -stp`. The arguments will first clone all the framework repos on the host, build the container and then setup the workspace, as well as a dedicated micromamba environment. Note that even though this process make take so time, it only needs to be run once. Additionally, all containers which use an IsaacSim image will also need a `-ngc $MY_NGC_KEY` argument, where `MY_NGC_KEY` should be set to your [nvidia NGC api-key](https://docs.nvidia.com/ngc/latest/ngc-user-guide.html#generating-api-key) to be able to pull IsaacSim docker images. The container will be dumped at `IBRIDO_CONTAINERS_PREFIX` with a `.sif` extension.

Note that IBRIDO's workspace and code are cloned on the host, mounted within the container and setup from within it. The container also comes with a *ibrido* micromamba environment, shipped with all necessary dependencies, Torch included.

### 3) Run container
- Navigate to the root folder corresponding to your chosen container (e.g. `./ibrido_u22/singularity`) and then run `export IBRIDO_CONTAINERS_PREFIX=$PWD`, if not already done.
- There are two ways to run the containers, an *interactive* and *detached* move:
    - For the interactive mode run:
        - `./run_interactive.sh`: you're now inside the container
        - `launch_byobu_ws.sh --cfg /root/ibrido_files/training_cfgs/$MY_CFG_FILE_PATH`: this will launch a [Byobu](https://www.byobu.org/) workspace (may take a couple of seconds) with multiple windows and all the necessary commands already preloaded on the right terminals. Commands are loaded based on the provided `MY_CFG_FILE_PATH` file, for instance `MY_CFG_FILE_PATH=centauro/training_cfg_centauro_cloop.sh`. The interactive mode is mainly useful for running single-environment evaluations and testing/developing the framework. 
    - For the detached mode run:
        - `./execute.sh --cfg $MY_CFG_FILE_PATH`, where `MY_CFG_FILE_PATH` should be a valid relative config path within the `ibrido_u*/singularity/files/training_cfgs/` folder. This is the preferred mode to run training on remote servers. It will launch all the core components of IBRIDO (world, MPC cluster and training scripts) and automatically start the training. Logs for each of the components are dumped separately at `$HOME/ibrido_logs/ibrido_run_$UNIQUE_ID`.

### 3) Run ablation studies
You can run ablation studies by executing `./execute_ablation.sh --cfg $MY_CFG_PATH`, where `MY_CFG_PATH` should be a valid relative directory path within the `ibrido_u*/singularity/files/training_cfgs/ablations/` folder, for instance `ablation_centauro_act_repeat_closed`. This will sequentially run a training for each config file found within `MY_CFG_PATH`.

### 4) Additional usage notes and HOWTOs
- Framework code, data, libraries, logs, etc...: everything will by default be located *on the HOST* at `$HOME/work/containers/$CONTAINER_NAME/`. You can change this location by editing the `BASE_FOLDER` variable in `$IBRIDO_CONTAINERS_PREFIX/files/bind_list.sh`.

- Shared memory *namespace*: one -- if not the most -- important argument to the whole framework, is the `--ns` (`SHM_NS` in the config files). You'll notice all scripts will have at least this argument exposed. This is a unique namespace employed by all shared memory components and allows for safe and efficient communication between all the framework's components. This also means that multiple independent containers can run concurrently without issues, as long as a different namespace was set in the configuration files. It is also possible to launch an instance of the framework in *detached* mode on the host and then connect to it afterwards by launching a new interactive session with the same namespace. In this case, just run components of IBRIDO which are NOT already running in the detached instance.

- When launching a training job on a remote server, it is often useful to run it as a background process and maintain the ability to reconnect to it at any time. To achieve this, we recommend starting the container inside a Byobu session on the host machine (e.g. `byobu new-session -n my_new_byobu_session`). This ensures that the session remains active even if your terminal disconnects. If you need to nest multiple Byobu sessions, note that you should press `Shift+F12` in the outer (root) session to disable its function keys, allowing the nested session to receive them properly.

- How to exit from an active interactive Byobu workspace: To close the preconfigured ibrido workspace you can run in any of the open terminals `byobu kill-session -t $BYOBU_SESSION_NAME`, where `BYOBU_SESSION_NAME` is the name of the Byobu session, usually shown in the bottom left corner of the workspace (e.g. for the ibrido_u20 container it's "ibrido_xbot").

- Enable mouse in interactive Byobu workspace: to allow clicking and navigating through terminals in the workspace using the mouse, press the `Alt+F12` (you'll see a `Mouse ON` message). 

- MPCs codegeneration: by default, all dynamics, costs, etc.. will be codegenerated to speed up MPC computations. Before running a training for the first time with a brand new MPC formulation, it is recommended to run the framework with `N_ENVS=1` and `REMOTE_STEPPING=0` in detached mode. You can inspect `$HOME/ibrido_logs/ibrido_run_$UNIQUE_ID/ibrido_rhc_cluster*.log` to check when the codegeneration is finished and then kill everything. This will dump all codegenerated functions into `$HOME/aux_data/MyClusterClientName_$SHM_NS/CodeGen/$SHM_NSRhc0`. After this, from within `$HOME/aux_data/MyClusterClientName_$SHM_NS/CodeGen/`, run `$HOME/ibrido_files/replicate_codegen.sh $SHM_NSRhc $N_COPIES` where `N_ENVS` should be equal to the number of environments you want to run during the training. If this is not done, then a separate codegen compilation process will be spawned for each controller and this, when the number of environments is high, can be very slow and also unnecessarily saturate the RAM of the host system. This feature is useful if running different MPC formulations within the same MPC cluster, but not necessary when running the same instance across the whole cluster.

### Acknowledgements
Thanks to [c-rizz](https://github.com/c-rizz) for the technical support in setting up the containers.
