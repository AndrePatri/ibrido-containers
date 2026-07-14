#!/bin/bash

PRESCIA_DT=10 # minutes
CHECK_INTERVAL=4  # Check interval in minutes
# Match the main launcher started by execute_container.sh; tighten with run token if provided.
# With no run token (the ablation case, where the config changes per iteration) fall back to the
# generic pattern, which matches whichever execute.sh is currently running.
#
# Signalling execute.sh is the right target: it traps SIGINT and forwards it to the singularity
# process group, which reaches launch_train_env.py's handler -> save model + dump the hdf5.
CFG="${1:-}"
RUN_TOKEN="${2:-}"
if [ -n "$RUN_TOKEN" ]; then
    TERMINATION_SCRIPT="execute.sh --cfg $CFG --run_token $RUN_TOKEN"
else
    TERMINATION_SCRIPT="execute.sh --cfg"
fi
TERMINATION_SIGNAL="SIGINT"

# Optional stop sentinel. An ablation runs its configs SEQUENTIALLY, so signalling the current run
# is not enough: without this, execute_ablation.sh would go straight on to the next config and start
# a fresh training that gets hard-killed at the walltime with nothing saved. execute_ablation.sh
# checks this file between iterations. Unset (e.g. for a single run) -> no-op.
STOP_FILE="${IBRIDO_STOP_FILE:-}"

source "$IBRIDO_CONTAINERS_PREFIX/franklin/slurm/utils.sh"

PRESCIA_DT_SEC=$(minutes_to_seconds "$PRESCIA_DT")
CHECK_INTERVAL_SEC=$(minutes_to_seconds "$CHECK_INTERVAL")

# Get total expected runtime in seconds
expected_runtime=$(get_walltime)
expected_runtime_sec=$(time_to_seconds "$expected_runtime") || expected_runtime_sec=0

if [ "$expected_runtime_sec" -le 0 ]; then
    echo "prescia_script: could not parse walltime ('$expected_runtime'); skipping early termination logic."
    exit 0
fi

# # Get the start time in seconds since epoch
start_time_sec=$(date +%s)

while true; do

    # Get the current time in seconds since epoch
    current_time_sec=$(date +%s)

    # Calculate the elapsed time
    elapsed_time_sec=$((current_time_sec - start_time_sec))

    # Calculate remaining time
    remaining_time_sec=$((expected_runtime_sec - elapsed_time_sec))

    # Check if remaining time is less than PRESCIA_DT_SEC
    if [ "$remaining_time_sec" -le "$PRESCIA_DT_SEC" ]; then
        # Stop first, signal second: set the sentinel BEFORE signalling, so that if the current run
        # exits quickly the ablation loop cannot slip in another iteration in between.
        if [ -n "$STOP_FILE" ]; then
            echo "prescia_script: setting stop sentinel at $STOP_FILE (no further runs will start)"
            touch "$STOP_FILE" 2>/dev/null || echo "prescia_script: WARNING could not create $STOP_FILE"
        fi
        echo "Sending termination signal to job..."
        send_signal_from_within "$TERMINATION_SCRIPT" "$TERMINATION_SIGNAL"
        exit 0
    else
        echo "Remaining time to nominal termination deadline: $remaining_time_sec s."
    fi

    # Sleep for the defined check interval before checking again
    sleep "$CHECK_INTERVAL_SEC"
done
