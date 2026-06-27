# Training Configs

u24 launchers use YAML configs. A config is required.

## Layout

- `common/runtime_defaults.yaml`: launcher defaults.
- `common/algorithms/`: training algorithm settings.
- `common/sites/`: optional machine settings for local, cluster, or onboard runs.
- `robots/<robot>/base.yaml`: robot description paths.
- `robots/<robot>/control/`: RHC, cluster, joint impedance, and control settings.
- `robots/<robot>/tasks/`: task, run, training, perception, and xacro settings.
- `robots/<robot>/backends/`: world interface settings.
- `runs/training/<robot>/`: full training configs.
- `runs/evaluation/sim_to_sim/`: saved-model evaluation in a simulator (isaac5x, genesis, or xmj). This
  covers both validating in the training simulator and cross-engine transfer -- they only differ by the
  `backends/*.yaml` you include (the `COMMENT_PREFIX`/`RNAME_SUFFIX` label the run).
- `runs/evaluation/sim_to_real/`: saved-model deployment through XBot/ZMQ.
- `runs/ablations/`: ablation configs.

The launchers resolve includes, apply command-line `--set VAR=VALUE` overrides, and record the resolved include stack in `IBRIDO_CFG_STACK`.

## Training

Training uses a full config under `runs/training/<robot>/`.

```bash
launch_training.sh --cfg runs/training/centauro/centauro_ub_cloop_isaac5x.yaml
```

Interactive Kyon smoke run:

```bash
launch_byobu_ws.sh --cfg runs/training/kyon02/kyon02_wheels_no_yaw_isaac5x.yaml --set N_ENVS=3 --set TOT_STEPS=1000
```

Use `--dry-run` to resolve the config and print launch commands without starting byobu, the world interface, the cluster, or training:

```bash
launch_byobu_ws.sh --cfg runs/training/kyon02/kyon02_wheels_no_yaw_isaac5x.yaml --set N_ENVS=3 --dry-run
```

## Saved-Model Evaluation

Saved-model evaluation uses `launch_bundle.sh`. The bundle provides the trained policy and saved training environment. The `--cfg` argument selects what should change for evaluation or deployment.

Sim-to-sim (any simulator). Validate in the training engine, or transfer to another one -- pick the
target by the `backends/*.yaml` in the config. For example a Talos policy across the three engines:

```bash
launch_bundle.sh --bundle /path/to/bundle.yaml --cfg runs/evaluation/sim_to_sim/talos/talos_isaac5x.yaml
launch_bundle.sh --bundle /path/to/bundle.yaml --cfg runs/evaluation/sim_to_sim/talos/talos_genesis.yaml
launch_bundle.sh --bundle /path/to/bundle.yaml --cfg runs/evaluation/sim_to_sim/talos/talos_xmj.yaml
```

A generic engine-agnostic override (keeps the bundle's task, only switches to eval) is also available:

```bash
launch_bundle.sh --bundle /path/to/bundle.yaml --cfg runs/evaluation/sim_to_sim/eval_sim_to_sim_isaac5x.yaml
```

Main profiles are under:

- `runs/evaluation/sim_to_sim/talos/`
- `runs/evaluation/sim_to_sim/kyon02/`
- `runs/evaluation/sim_to_sim/centauro/`

Sim-to-real rehearsal through XBot/ZMQ:

```bash
launch_bundle.sh --bundle /path/to/bundle.yaml --cfg runs/evaluation/sim_to_real/centauro/centauro_ub_cloop_rt_sim.yaml
launch_bundle.sh --bundle /path/to/bundle.yaml --cfg runs/evaluation/sim_to_real/kyon02/kyon02_wheels_no_yaw_rt_sim.yaml
```

RT-sim profiles are under:

- `runs/evaluation/sim_to_real/kyon02/*_rt_sim.yaml`
- `runs/evaluation/sim_to_real/centauro/*_rt_sim.yaml`
- `runs/evaluation/sim_to_real/b2w/b2w_rt_sim.yaml`

Real robot style XBot/ZMQ deployment:

```bash
launch_bundle.sh --bundle /path/to/bundle.yaml --cfg runs/evaluation/sim_to_real/kyon02/kyon02_wheels_no_yaw_rt_real.yaml
launch_bundle.sh --bundle /path/to/bundle.yaml --cfg runs/evaluation/sim_to_real/centauro/centauro_ub_cloop_rt_real.yaml
```

RT-real profiles are provided for the main Kyon02 and Centauro closed-loop controllers. Wheel profiles use the continuous-joint controller variants.

`launch_bundle.sh` protects contract variables from accidental `--set` overrides unless `--allow_contract_override` is passed.

## Custom Args

`custom_args` may be grouped by category for readability. The loader flattens them before calling the existing launch scripts:

```yaml
custom_args:
  control:
    use_jnt_v_feedback:
      dtype: bool
      value: true
  robot_description:
    add_upper_body:
      dtype: bool
      value: true
```

## Ablations

```bash
execute_ablation.sh --cfg_dir runs/ablations/ablation_centauro_act_repeat_closed
execute_ablation.sh --cfg_dir runs/ablations --recursive --dry-run
```
