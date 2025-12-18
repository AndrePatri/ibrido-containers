# Utility helpers adapted from PBS-based helpers to Slurm equivalent

# Send a signal to a job (prefer scancel --signal if available)
function send_signal() {
local job_id=$1
local script_pattern=$2
local signal=${3:-SIGINT}


if [ -z "$job_id" ] || [ -z "$script_pattern" ]; then
echo "Usage: send_signal <jobid> <script_pattern> [SIGNAL]"
return 1
fi


# Prefer Slurm's scancel --signal mechanism (requires slurm version with --signal support)
if command -v scancel >/dev/null 2>&1; then
# Try sending the signal to the job
if scancel --signal="$signal" "$job_id" 2>/dev/null; then
echo "Sent $signal to job $job_id via scancel."
return 0
fi
fi


# If scancel didn't do it (or to act on processes directly), try srun inside allocation
if command -v srun >/dev/null 2>&1; then
if srun --jobid="$job_id" --ntasks=1 bash -lc "pkill -$signal -f '$script_pattern'" 2>/dev/null; then
echo "Sent $signal to processes matching '$script_pattern' inside job $job_id via srun."
return 0
fi
fi


# Fallback: SSH to node and kill matching procs
local node=$(squeue -j "$job_id" -h -o "%N" | cut -d',' -f1)
if [ -z "$node" ]; then
echo "Could not determine the execution host for job $job_id."
return 1
fi


echo "Preparing to send signal from within job $job_id on node $node"
ssh -t "$node" "
pids=\$(pgrep -f '${script_pattern}')
echo 'Matching PIDs:' \"\$pids\"
if [ -z \"\$pids\" ]; then
echo 'No processes found matching pattern \"${script_pattern}\"'
exit 1
fi
for pid in \$pids; do
kill -s \"$signal\" \"\$pid\"
done
"
}


alias send_signal_outside=send_signal


# Get walltime (TimeLimit) for the current Slurm job
function get_walltime() {
if [ -z "$SLURM_JOB_ID" ]; then
echo "Error: SLURM_JOB_ID environment variable is not set."
return 1
fi


local walltime=$(scontrol show job "$SLURM_JOB_ID" | awk -F"TimeLimit=" '/TimeLimit/ {print $2}' | awk '{print $1}')
echo "$walltime"
}
alias get_walltime=get_walltime


# Convert h:m:s to seconds
function time_to_seconds() {
local time=$1
local h=$(echo $time | cut -d':' -f1)
local m=$(echo $time | cut -d':' -f2)
local s=$(echo $time | cut -d':' -f3)
echo $((10#$h*3600 + 10#$m*60 + 10#$s))
}


# Convert minutes to seconds
function minutes_to_seconds() {
local minutes=$1
echo $((minutes * 60))
}
alias minutes_to_seconds="minutes_to_seconds"
