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
