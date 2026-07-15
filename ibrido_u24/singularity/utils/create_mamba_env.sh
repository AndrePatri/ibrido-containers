#!/bin/bash
set -e # exiting if any cmd fails

source /root/ibrido_utils/mamba_utils/bin/_activate_current_env.sh # enable mamba for this shell

# fallbacks so this works on images built before the genesis env vars were added to the .def
# (the ymls are bind-mounted from files/, so they are current without a full .sif rebuild).
MAMBA_ENV_NAME="${MAMBA_ENV_NAME:-ibrido}"
MAMBA_ENV_NAME_ISAAC="${MAMBA_ENV_NAME_ISAAC:-ibrido_isaac_py11}"
MAMBA_ENV_NAME_GENESIS="${MAMBA_ENV_NAME_GENESIS:-ibrido_genesis}"
MAMBA_ENV_FPATH="${MAMBA_ENV_FPATH:-${HOME}/ibrido_files/mamba_env.yml}"
MAMBA_ENV_FPATH_ISAAC="${MAMBA_ENV_FPATH_ISAAC:-${HOME}/ibrido_files/mamba_env_isaac_py11.yml}"
MAMBA_ENV_FPATH_GENESIS="${MAMBA_ENV_FPATH_GENESIS:-${HOME}/ibrido_files/mamba_env_genesis.yml}"

# Re-create envs idempotently: remove only the env dir (envs/<name>) and recreate. This NEVER
# touches the package cache (${MAMBA_ROOT_PREFIX}/pkgs), so re-setups reuse already-downloaded
# packages/repodata instead of re-downloading everything, and a partial/previous setup does not
# abort the run with "environment already exists".
recreate_env() {
    local env_name="$1"
    local env_fpath="$2"
    echo "Re-creating ${env_name} environment from ${env_fpath} (package cache preserved)..."
    micromamba env remove -y -n "${env_name}" >/dev/null 2>&1 || true
    # default log level (no --log-level error) so the solve + download/extract progress is visible
    # instead of going silent during the long solve.
    micromamba env create -y -f "${env_fpath}"
}

# Isaac is created LAST on purpose: it is the most fragile env (Isaac Sim 5.1 pins python 3.11 and
# is picky about torch), and with `set -e` a failure here aborts the script. Building main and
# genesis first means a partial/failed Isaac setup still leaves a usable container for the base and
# genesis backends (genesis training does not touch the Isaac env at all).
recreate_env "${MAMBA_ENV_NAME}"         "${MAMBA_ENV_FPATH}"
recreate_env "${MAMBA_ENV_NAME_GENESIS}" "${MAMBA_ENV_FPATH_GENESIS}"
recreate_env "${MAMBA_ENV_NAME_ISAAC}"   "${MAMBA_ENV_FPATH_ISAAC}"
