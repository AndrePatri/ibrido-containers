#!/bin/bash
set +e # do not exit if any cmd fails
echo '--> Updating workspace code...'

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_folder="$(dirname "$THIS_DIR")"

source "${root_folder}/files/bind_list.sh"

cd $IBRIDO_WS_SRC

# Iterate through IBRIDO_GITDIRS
for ((i = 0; i < ${#IBRIDO_GITDIRS[@]}; i++)); do
    git_url="${IBRIDO_GITDIRS[$i]}"
    
    # Extract the repo name by stripping off the URL prefix and ".git" suffix
    repo_name=$(basename "${git_url%%.git*}" | cut -d'*' -f1)
    
    # Determine the directory path
    repo_path="$IBRIDO_WS_SRC/$repo_name"

    # Check if the directory exists and perform a git pull
    if [ -d "$repo_path" ]; then
        echo "Pulling latest changes in $repo_name..."
        cd "$repo_path"
        git pull &
        cd "$IBRIDO_WS_SRC"
    else
        echo "Directory $repo_path does not exist. Skipping..."
    fi
done
wait
echo 'update done.'

set -e