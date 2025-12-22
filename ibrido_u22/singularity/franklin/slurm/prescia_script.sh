#!/bin/bash

PRESCIA_DT=10 # minutes
CHECK_INTERVAL=4  # Check interval in minutes
# Match the main launcher started by execute_container.sh; tighten with run token if provided
RUN_TOKEN="$1"
if [ -n "$RUN_TOKEN" ]; then
    TERMINATION_SCRIPT="execute.sh --run_token $RUN_TOKEN"
else
    TERMINATION_SCRIPT="execute.sh --cfg"
fi
TERMINATION_SIGNAL="SIGINT"

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
        echo "Sending termination signal to job..."
        send_signal_from_within "$TERMINATION_SCRIPT" "$TERMINATION_SIGNAL"
        exit 0
    else
        echo "Remaining time to nominal termination deadline: $remaining_time_sec s."
    fi

    # Sleep for the defined check interval before checking again
    sleep "$CHECK_INTERVAL_SEC"
done
