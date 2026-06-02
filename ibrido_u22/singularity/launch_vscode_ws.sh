#!/bin/bash
set -e

if [ -z "$IBRIDO_CONTAINERS_PREFIX" ]; then
    echo "IBRIDO_CONTAINERS_PREFIX variable has not been seen. Please set it to \${path_to_ibrido-containers}/ibrido_u22/singularity."
    exit 1
fi

source "${IBRIDO_CONTAINERS_PREFIX}/files/bind_list.sh"

usage() {
    cat <<'EOF'
Usage: launch_vscode_ws.sh [options]

Start a VS Code Remote Tunnel from inside the IBRIDO Apptainer container.
This keeps VS Code/Pylance in the same environment where /root/ibrido_ws,
micromamba, and editable installs are valid.

Options:
  -s, --use_sudo             Run singularity through sudo.
      --name <name>          Stable tunnel name, max 20 chars. Default: ib-u22-$USER.
      --random-name          Let VS Code choose a random tunnel name.
      --download-only        Download/update the standalone VS Code CLI, then exit.
      --force-download       Re-download the standalone VS Code CLI.
      --status               Print tunnel status from inside the container.
      --kill                 Stop a running tunnel from inside the container.
      --no-install-extensions
                             Do not preinstall Python/Pylance extensions.
      --cli-url <url>        Override VS Code CLI download URL.
  -h, --help                 Show this help.

The first tunnel launch requires network access and VS Code/GitHub login.
EOF
}

use_sudo=false
random_name=false
download_only=false
force_download=false
install_extensions=true
tunnel_subcommand=""
tunnel_name="ib-u22-${USER:-$(whoami)}"
if [ -n "$VSCODE_CLI_URL" ]; then
    vscode_cli_url="$VSCODE_CLI_URL"
elif command -v code >/dev/null 2>&1; then
    host_code_version="$(code --version | sed -n '1p')"
    vscode_cli_url="https://update.code.visualstudio.com/${host_code_version}/cli-linux-x64/stable"
else
    vscode_cli_url="https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64"
fi

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -s|--use_sudo)
            use_sudo=true
            ;;
        --name)
            tunnel_name="$2"
            shift
            ;;
        --random-name)
            random_name=true
            ;;
        --download-only)
            download_only=true
            ;;
        --force-download)
            force_download=true
            ;;
        --status)
            tunnel_subcommand="status"
            ;;
        --kill)
            tunnel_subcommand="kill"
            ;;
        --no-install-extensions)
            install_extensions=false
            ;;
        --cli-url)
            vscode_cli_url="$2"
            shift
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

if [ "$random_name" = false ] && [[ ! "$tunnel_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Tunnel name can only contain letters, numbers, dot, underscore, and dash."
    exit 1
fi

if [ "$random_name" = false ] && [ "${#tunnel_name}" -gt 20 ]; then
    echo "Tunnel name '${tunnel_name}' is ${#tunnel_name} characters long; VS Code tunnel names must be at most 20 characters."
    echo "Use --name <short-name>, for example: --name ib-u22"
    exit 1
fi

vscode_root="${IBRIDO_PREFIX}/vscode"
vscode_cli_dir="${vscode_root}/code-cli"
vscode_cli="${vscode_cli_dir}/code"
vscode_cli_archive="${vscode_root}/vscode_cli.tar.gz"

download_vscode_cli() {
    if [ "$force_download" = false ] && [ -x "$vscode_cli" ]; then
        return
    fi

    echo "--> Downloading standalone VS Code CLI..."
    mkdir -p "$vscode_cli_dir"
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' RETURN

    if command -v curl >/dev/null 2>&1; then
        curl -Lk "$vscode_cli_url" --output "$vscode_cli_archive"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$vscode_cli_archive" "$vscode_cli_url"
    else
        echo "Neither curl nor wget is available. Download the VS Code CLI manually to:"
        echo "  $vscode_cli"
        exit 1
    fi

    tar -xf "$vscode_cli_archive" -C "$tmpdir"
    found_code="$(find "$tmpdir" -type f -name code | head -n 1)"
    if [ -z "$found_code" ]; then
        echo "Could not find a code executable in the downloaded VS Code CLI archive."
        exit 1
    fi

    tmp_code="${vscode_cli}.new"
    cp "$found_code" "$tmp_code"
    chmod +x "$tmp_code"
    mv -f "$tmp_code" "$vscode_cli"
}

download_vscode_cli

if [ "$download_only" = true ]; then
    "$vscode_cli" --version
    exit 0
fi

mkdir -p "${vscode_root}/cli-data" "${vscode_root}/server-data" "${vscode_root}/extensions"

IFS=','
binddirs="${IBRIDO_B_ALL[*]},${vscode_root}:/root/ibrido_vscode:rw"
unset IFS

tunnel_args=(
    /root/ibrido_vscode/code-cli/code
    tunnel
    --cli-data-dir /root/ibrido_vscode/cli-data
)

if [ -n "$tunnel_subcommand" ]; then
    tunnel_args+=("$tunnel_subcommand")
else
    tunnel_args+=(
        --server-data-dir /root/ibrido_vscode/server-data
        --extensions-dir /root/ibrido_vscode/extensions
        --accept-server-license-terms
    )

    if [ "$random_name" = true ]; then
        tunnel_args+=(--random-name)
    else
        tunnel_args+=(--name "$tunnel_name")
    fi

    if [ "$install_extensions" = true ]; then
        tunnel_args+=(--install-extension ms-python.python)
        tunnel_args+=(--install-extension ms-python.vscode-pylance)
    fi
fi

container_cmd='
set -e
source /root/ibrido_utils/mamba_utils/bin/_activate_current_env.sh
micromamba activate "${MAMBA_ENV_NAME}"
source /root/ibrido_ws/setup.bash 2>/dev/null || true
cd /root/ibrido_ws
exec "$@"
'

if [ -n "$tunnel_subcommand" ]; then
    echo "--> Running VS Code tunnel '${tunnel_subcommand}' inside IBRIDO container..."
else
    echo "--> Starting VS Code tunnel inside IBRIDO container..."
    echo "--> Workspace inside tunnel: /root/ibrido_ws"
fi

if $use_sudo; then
    cd /
    sudo singularity exec \
        --bind "$binddirs" \
        --no-mount home,cwd \
        --nv "$IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif" \
        bash -lc "$container_cmd" bash "${tunnel_args[@]}"
else
    cd /
    singularity exec \
        --bind "$binddirs" \
        --no-mount home,cwd \
        --nv "$IBRIDO_CONTAINERS_PREFIX/ibrido_isaac.sif" \
        bash -lc "$container_cmd" bash "${tunnel_args[@]}"
fi
