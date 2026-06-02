#!/bin/bash
set -e

if [ -z "$IBRIDO_CONTAINERS_PREFIX" ]; then
    echo "IBRIDO_CONTAINERS_PREFIX variable has not been seen. Please set it to \${path_to_ibrido-containers}/ibrido_u22/singularity."
    exit 1
fi

source "${IBRIDO_CONTAINERS_PREFIX}/files/bind_list.sh"

usage() {
    cat <<'EOF'
Usage: launch_vscode_ssh_ws.sh [options]

Start an SSH server inside the IBRIDO Apptainer container and open the IBRIDO
workspace in VS Code Desktop through Remote-SSH. This avoids vscode.dev,
VS Code Remote Tunnels, and GitHub/Microsoft tunnel login prompts.

Options:
  -s, --use_sudo        Run singularity through sudo.
      --port <port>     Localhost SSH port. Default: 2222.
      --host <name>     SSH config host alias. Default: ibrido-u22.
      --foreground      Keep sshd in the foreground and print connection info.
      --no-open-code    Start/check SSH only; do not open VS Code Desktop.
      --no-install-extensions
                       Do not automatically install VS Code extensions.
      --status          Print whether the generated sshd PID is alive.
      --kill            Stop the generated sshd process.
      --print-config    Print the generated SSH config and exit.
      --install-config Add an Include entry to ~/.ssh/config, then exit.
      --no-install-config
                       Do not add the generated Include entry before launch.
      --check          Validate the container sshd configuration, then exit.
  -h, --help           Show this help.

Default launch starts sshd in the background, waits until SSH is reachable, and
then runs VS Code with a Remote-SSH workspace URI.
EOF
}

use_sudo=false
port="${IBRIDO_VSCODE_SSH_PORT:-2222}"
host_alias="${IBRIDO_VSCODE_SSH_HOST:-ibrido-u22}"
status_only=false
kill_server=false
print_config=false
install_config=false
install_config_on_launch=true
check_only=false
foreground=false
open_code=true
install_extensions=true

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -s|--use_sudo)
            use_sudo=true
            ;;
        --port)
            port="$2"
            shift
            ;;
        --host)
            host_alias="$2"
            shift
            ;;
        --foreground)
            foreground=true
            open_code=false
            ;;
        --no-open-code)
            open_code=false
            ;;
        --no-install-extensions)
            install_extensions=false
            ;;
        --status)
            status_only=true
            ;;
        --kill)
            kill_server=true
            ;;
        --print-config)
            print_config=true
            ;;
        --install-config)
            install_config=true
            ;;
        --no-install-config)
            install_config_on_launch=false
            ;;
        --check)
            check_only=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

if [[ ! "$port" =~ ^[0-9]+$ ]] || [ "$port" -lt 1024 ] || [ "$port" -gt 65535 ]; then
    echo "Port must be an integer between 1024 and 65535."
    exit 1
fi

if [[ ! "$host_alias" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Host alias can only contain letters, numbers, dot, underscore, and dash."
    exit 1
fi

ssh_root="${IBRIDO_PREFIX}/vscode-ssh"
ssh_dir="${ssh_root}/ssh"
etc_dir="${ssh_root}/etc"
run_dir="${ssh_root}/run_sshd"
vscode_server_dir="${ssh_root}/vscode-server"
sshd_config="${ssh_root}/sshd_config"
ssh_config="${ssh_root}/ssh_config"
host_key="${ssh_dir}/ssh_host_ed25519_key"
client_key="${ssh_dir}/ibrido_vscode_key"
authorized_keys="${ssh_dir}/authorized_keys"
pid_file="${run_dir}/sshd.pid"
log_file="${ssh_root}/sshd.log"
remote_workspace="/root/ibrido_files/ibrido_u22.code-workspace"
workspace_uri="vscode-remote://ssh-remote+${host_alias}${remote_workspace}"

prepare_ssh_files() {
    mkdir -p "$ssh_dir" "$etc_dir" "$run_dir" "$vscode_server_dir"
    chmod 700 "$ssh_dir"

    if [ ! -f "$host_key" ]; then
        ssh-keygen -q -t ed25519 -N '' -f "$host_key"
    fi

    if [ ! -f "$client_key" ]; then
        ssh-keygen -q -t ed25519 -N '' -f "$client_key"
    fi

    if [ ! -f "${client_key}.pub" ]; then
        ssh-keygen -y -f "$client_key" > "${client_key}.pub"
    fi
    cp "${client_key}.pub" "$authorized_keys"
    chmod 600 "$host_key" "$client_key" "$authorized_keys"

    cat > "${etc_dir}/passwd" <<EOF
root:x:0:0:root:/root:/bin/bash
ubuntu:x:1000:1000::/root:/bin/bash
EOF

    cat > "${etc_dir}/shadow" <<EOF
root:*:19646:0:99999:7:::
ubuntu::19646:0:99999:7:::
EOF

    cat > "${etc_dir}/group" <<EOF
root:x:0:
ubuntu:x:1000:
EOF

    chmod 644 "${etc_dir}/passwd" "${etc_dir}/group"
    chmod 600 "${etc_dir}/shadow"

    cat > "$sshd_config" <<EOF
Port ${port}
ListenAddress 127.0.0.1
HostKey /root/.ssh/ssh_host_ed25519_key
AuthorizedKeysFile /root/.ssh/authorized_keys
PidFile /run/sshd/sshd.pid
PasswordAuthentication no
KbdInteractiveAuthentication no
ChallengeResponseAuthentication no
PubkeyAuthentication yes
PermitRootLogin no
UsePAM no
AllowUsers ubuntu
X11Forwarding no
AllowTcpForwarding yes
PrintMotd no
Subsystem sftp /usr/lib/openssh/sftp-server
LogLevel VERBOSE
EOF

    cat > "$ssh_config" <<EOF
Host ${host_alias}
    HostName 127.0.0.1
    Port ${port}
    User ubuntu
    IdentityFile ${client_key}
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
    UserKnownHostsFile ${ssh_dir}/known_hosts
EOF
}

print_connection_info() {
    echo "--> SSH config: ${ssh_config}"
    echo "--> VS Code Desktop connection:"
    echo "    1. Install the Remote - SSH extension if needed."
    echo "    2. This script opens the remote workspace with:"
    echo "       code --file-uri ${workspace_uri}"
    echo "    3. If opening manually, run 'Remote-SSH: Connect to Host...' and select:"
    echo "       ${host_alias}"
    echo "       then open folder:"
    echo "       /root/ibrido_ws"
    echo ""
    echo "Equivalent SSH command:"
    echo "    ssh -F ${ssh_config} ${host_alias}"
}

prepare_ssh_files

install_user_config() {
    user_ssh_dir="${HOME}/.ssh"
    user_ssh_config="${user_ssh_dir}/config"
    include_line="Include ${ssh_config}"
    tmp_config="${user_ssh_config}.ibrido_tmp.$$"

    mkdir -p "$user_ssh_dir"
    chmod 700 "$user_ssh_dir"
    touch "$user_ssh_config"
    chmod 600 "$user_ssh_config"

    grep -Fxv "# IBRIDO VS Code Remote-SSH" "$user_ssh_config" \
        | grep -Fxv "$include_line" > "$tmp_config" || true
    {
        printf '# IBRIDO VS Code Remote-SSH\n%s\n\n' "$include_line"
        cat "$tmp_config"
    } > "$user_ssh_config"
    rm -f "$tmp_config"
    echo "Installed top-level SSH config include for ${ssh_config} in ${user_ssh_config}."
}

if [ "$install_config" = true ]; then
    install_user_config
    print_connection_info
    exit 0
fi

if [ "$status_only" = true ]; then
    if [ -f "$pid_file" ] && ps -p "$(cat "$pid_file")" >/dev/null 2>&1; then
        echo "IBRIDO VS Code SSH server is running with PID $(cat "$pid_file")."
        exit 0
    fi
    echo "IBRIDO VS Code SSH server is not running."
    exit 1
fi

if [ "$kill_server" = true ]; then
    if [ -f "$pid_file" ] && ps -p "$(cat "$pid_file")" >/dev/null 2>&1; then
        sshd_pid="$(cat "$pid_file")"
        kill -TERM "$sshd_pid"
        echo "Stopped IBRIDO VS Code SSH server with PID ${sshd_pid}."
    else
        echo "IBRIDO VS Code SSH server is not running."
    fi
    exit 0
fi

if [ "$print_config" = true ]; then
    print_connection_info
    exit 0
fi

IFS=','
binddirs="${IBRIDO_B_ALL[*]},${ssh_dir}:/root/.ssh:rw,${etc_dir}/passwd:/etc/passwd:ro,${etc_dir}/shadow:/etc/shadow:ro,${etc_dir}/group:/etc/group:ro,${run_dir}:/run/sshd:rw,${vscode_server_dir}:/root/.vscode-server:rw"
unset IFS

container_cmd='
set -e
source /root/ibrido_utils/mamba_utils/bin/_activate_current_env.sh
micromamba activate "${MAMBA_ENV_NAME}"
source /root/ibrido_ws/setup.bash 2>/dev/null || true
/usr/sbin/sshd -t -f /root/.ssh/sshd_config
exec /usr/sbin/sshd -D -e -f /root/.ssh/sshd_config
'

cp "$sshd_config" "${ssh_dir}/sshd_config"

server_running=false
if [ -f "$pid_file" ] && ps -p "$(cat "$pid_file")" >/dev/null 2>&1; then
    server_running=true
fi

if [ "$check_only" = true ]; then
    echo "--> Validating VS Code Remote-SSH server config inside IBRIDO container..."
    if $use_sudo; then
        cd /
        sudo singularity exec \
            --bind "$binddirs" \
            --no-mount home,cwd \
            --nv "$IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif" \
            bash -lc "/usr/sbin/sshd -t -f /root/.ssh/sshd_config"
    else
        cd /
        singularity exec \
            --bind "$binddirs" \
            --no-mount home,cwd \
            --nv "$IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif" \
            bash -lc "/usr/sbin/sshd -t -f /root/.ssh/sshd_config"
    fi
    echo "Container sshd configuration is valid."
    exit 0
fi

if [ "$install_config_on_launch" = true ]; then
    install_user_config
fi

wait_for_ssh() {
    for _ in $(seq 1 45); do
        if ssh -F "$ssh_config" \
            -o BatchMode=yes \
            -o ConnectTimeout=2 \
            "$host_alias" "test -f '${remote_workspace}'" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

open_vscode_workspace() {
    if ! command -v code >/dev/null 2>&1; then
        echo "VS Code CLI 'code' was not found on PATH."
        echo "Open this URI manually from VS Code Desktop:"
        echo "  ${workspace_uri}"
        return 1
    fi

    echo "--> Opening VS Code Desktop remote workspace..."
    code --file-uri "$workspace_uri"
}

install_vscode_extensions() {
    if [ "$install_extensions" != true ]; then
        return 0
    fi

    if ! command -v code >/dev/null 2>&1; then
        echo "VS Code CLI 'code' was not found on PATH; skipping automatic extension installation."
        return 0
    fi

    echo "--> Ensuring VS Code Remote-SSH extension is installed locally..."
    if ! code --install-extension ms-vscode-remote.remote-ssh >/dev/null 2>&1; then
        echo "Warning: could not install ms-vscode-remote.remote-ssh locally."
    fi

    echo "--> Ensuring Python and Pylance extensions are installed on ${host_alias}..."
    for extension in ms-python.python ms-python.vscode-pylance; do
        if ! code --remote "ssh-remote+${host_alias}" --install-extension "$extension" >/dev/null 2>&1; then
            echo "Warning: could not install ${extension} on ${host_alias}."
        fi
    done
}

echo "--> Starting VS Code Remote-SSH server inside IBRIDO container..."
echo "--> SSH endpoint: ${host_alias} on 127.0.0.1:${port}"
echo "--> Workspace inside SSH session: /root/ibrido_ws"
print_connection_info
echo "--> Logs will also be written to: ${log_file}"

if [ "$server_running" = true ]; then
    echo "--> IBRIDO VS Code SSH server is already running with PID $(cat "$pid_file")."
    if [ "$foreground" = true ]; then
        echo "Use --kill first if you want to restart it in the foreground."
        exit 1
    fi
else
if [ "$foreground" = true ]; then
    if $use_sudo; then
        cd /
        sudo singularity exec \
            --bind "$binddirs" \
            --no-mount home,cwd \
            --nv "$IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif" \
            bash -lc "$container_cmd" 2>&1 | tee "$log_file"
    else
        cd /
        singularity exec \
            --bind "$binddirs" \
            --no-mount home,cwd \
            --nv "$IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif" \
            bash -lc "$container_cmd" 2>&1 | tee "$log_file"
    fi
    exit $?
fi

    if $use_sudo; then
        cd /
        setsid sudo singularity exec \
            --bind "$binddirs" \
            --no-mount home,cwd \
            --nv "$IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif" \
            bash -lc "$container_cmd" >"$log_file" 2>&1 &
    else
        cd /
        setsid singularity exec \
            --bind "$binddirs" \
            --no-mount home,cwd \
            --nv "$IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif" \
            bash -lc "$container_cmd" >"$log_file" 2>&1 &
    fi

    launcher_pid=$!
    echo "--> Container launcher PID: ${launcher_pid}"
fi

if ! wait_for_ssh; then
    echo "Timed out waiting for SSH endpoint ${host_alias} on 127.0.0.1:${port}."
    echo "Check logs at: ${log_file}"
    exit 1
fi

echo "--> SSH endpoint is reachable."

if [ "$open_code" = true ]; then
    install_vscode_extensions
    open_vscode_workspace
fi
