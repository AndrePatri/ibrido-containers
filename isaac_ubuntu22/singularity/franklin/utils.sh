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
        echo "Retrieved PIDS: \$pids"
        if [ -z \"\$pids\" ]; then
            echo 'No processes found matching pattern \"${script_pattern}\"'
            exit 1
        fi

        # Loop through each PID and send the specified signal
        for pid in \$pids; do
            echo 'Sending signal ${signal} to process with PID: \$pid'
            kill -${signal} \$pid
            echo 'Done.'
        done

        # Open a bash shell
        bash
    "
}
alias send_signal="send_signal"
