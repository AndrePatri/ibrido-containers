function qifun() {
             ssh $(qstat -f $1 | grep exec_host | cut -f 7 -d " " | cut -f 1 -d "/") cat /var/spool/pbs/spool/$1*
}

alias print_out="qifun"


function qhfun() {
             ssh -t $(qstat -f $1 | grep exec_host | cut -f 7 -d " " | cut -f 1 -d "/") htop -u $USER
}

alias cpu_usage="qhfun"

function qnfun() {
             ssh -t $(qstat -f $1 | grep exec_host | cut -f 7 -d " " | cut -f 1 -d "/") nvidia-smi
}

alias gpu_usage="qnfun"

function launch_bash_session() {
    if [ -z "$1" ]; then
        echo "Usage: launch_bash_session <job_id>"
        return 1
    fi

    # Get the execution host of the job
    exec_host=$(qstat -f $1 | grep exec_host | cut -f 7 -d " " | cut -f 1 -d "/")

    if [ -z "$exec_host" ]; then
        echo "Could not determine the execution host for job $1."
        return 1
    fi

    echo "Launching bash session on host: $exec_host"
    ssh -t $exec_host bash
}
alias bash_session="launch_bash_session"

function send_signal() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Usage: send_signal <job_id> <script_pattern> [<signal>]"
        return 1
    fi

    local job_id=$1
    local script_pattern=$2
    local signal=${3:-SIGINT}  # Default to SIGINT if no signal is provided

    # Get the execution host of the job
    local exec_host=$(qstat -f $job_id | grep exec_host | cut -f 7 -d " " | cut -f 1 -d "/")

    if [ -z "$exec_host" ]; then
        echo "Could not determine the execution host for job $job_id."
        return 1
    fi

    ssh -t "$exec_host" "
        # Find the PIDs of the scripts matching the pattern
        pids=\$(pgrep -f '${script_pattern}')
        echo "Matching PIDS: \$pids"
        if [ -z \"\$pids\" ]; then
            echo 'No processes found matching pattern \"${script_pattern}\"'
            exit 1
        fi

        # Loop through each PID and send the specified signal
        for pid in \$pids; do
            kill -s ${signal} \$pid
        done

        # Open a bash shell
        bash
    "
}
alias send_signal="send_signal"

get_walltime() {
    # Check if SCHED_JOBID environment variable is set
    if [ -z "$SCHED_JOBID" ]; then
        echo "Error: SCHED_JOBID environment variable is not set."
        return 1
    fi

    # Run the qstat command and parse the JSON output using jq
    walltime=$(qstat -f "$SCHED_JOBID" -F json | jq -r '.Jobs[].Resource_List.walltime')

    # Check if jq command was successful
    if [ $? -ne 0 ]; then
        echo "Error: Unable to parse the JSON output."
        return 1
    fi

    echo "$walltime"
}
alias get_walltime="get_walltime"

send_signal_from_within() {
    # Check if SCHED_JOBID and script_pattern are provided

    local script_pattern=$1
    local signal=${2:-SIGINT}  # Default to SIGINT if no signal is provided

    # Extract the job ID before the first dot
    local job_id=$(echo "$SCHED_JOBID" | cut -d'.' -f1)
    echo "Preparing to send signal from within job $job_id"

    # Find the PIDs of the scripts matching the pattern
    pids=$(pgrep -f "$script_pattern")
    echo "Matching PIDs: $pids"
    if [ -z "$pids" ]; then
        echo "No processes found matching pattern \"$script_pattern\""
        return 1
    fi

    # Loop through each PID and send the specified signal
    for pid in $pids; do
        kill -s "$signal" "$pid"
    done
}
alias send_signal_from_within="send_signal_from_within"

# Convert h:m:s to seconds
time_to_seconds() {
    local time=$1
    local h=$(echo $time | cut -d':' -f1)
    local m=$(echo $time | cut -d':' -f2)
    local s=$(echo $time | cut -d':' -f3)
    echo $((10#$h*3600 + 10#$m*60 + 10#$s))
}

# Convert minutes to seconds
minutes_to_seconds() {
    local minutes=$1
    echo $((minutes * 60))
}
alias minutes_to_seconds="minutes_to_seconds"
