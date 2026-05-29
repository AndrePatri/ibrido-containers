# Training Configs

u24 launchers use YAML configs. A config is required.

```bash
launch_training.sh --cfg runs/centauro_ub_cloop_isaac5x.yaml
launch_byobu_ws.sh --cfg runs/centauro_ub_cloop_isaac5x.yaml --set N_ENVS=3
```

`--dry-run` resolves the config, prints the launch commands, and writes launch metadata. It does not start byobu, the world interface, the cluster, or the training environment.

The resolved include stack is recorded in `IBRIDO_CFG_STACK` and copied to `run_metadata/configs/`.

## Layout

- `common/runtime_defaults.yaml`: launcher defaults.
- `common/algorithms/`: training algorithm settings.
- `common/sites/`: optional machine settings for local, cluster, or onboard runs.
- `robots/<robot>/base.yaml`: robot description paths.
- `robots/<robot>/control/`: RHC, cluster, joint impedance, and control settings.
- `robots/<robot>/tasks/`: task, run, training, perception, and xacro settings.
- `robots/<robot>/backends/`: world interface settings.
- `runs/`: launchable training and evaluation configs.
- `runs/ablations/`: launchable ablation configs.
- `transfers/`: bundle replay configs.

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

## Bundle Transfer

Saved runs are replayed with an explicit transfer config:

```bash
launch_bundle.sh --bundle /path/to/bundle.yaml --cfg transfers/eval_same_domain_isaac5x.yaml --dry-run
launch_bundle.sh --bundle /path/to/bundle.yaml --cfg transfers/eval_cross_sim_xmj.yaml --dry-run
launch_bundle.sh --bundle /path/to/bundle.yaml --cfg transfers/rt_sim_xbot_zmq.yaml --dry-run
launch_bundle.sh --bundle /path/to/bundle.yaml --cfg transfers/rt_real_xbot_zmq.yaml --dry-run
```

`launch_bundle.sh` restores the saved training environment, then applies the transfer config. Contract variables are protected from `--set` unless `--allow_contract_override` is passed.

## Ablations

```bash
execute_ablation.sh --cfg_dir runs/ablations/ablation_centauro_act_repeat_closed
execute_ablation.sh --cfg_dir runs/ablations --recursive --dry-run
```
