#!/bin/bash
set +e # do not exit if any cmd fails
echo '--> Updating workspace code...'

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_folder="$(dirname "$THIS_DIR")"

source "${root_folder}/files/bind_list.sh"

cd "$IBRIDO_WS_SRC" || { echo "Could not cd to $IBRIDO_WS_SRC"; exit 1; }

# Iterate through parsed git sources (these are created in bind_list.sh)
for ((i = 0; i < ${#IBRIDO_GIT_SRC[@]}; i++)); do
    git_url="${IBRIDO_GIT_SRC[$i]}"
    custom_dir="${IBRIDO_GIT_DIR[$i]}"  # may be empty

    # If a custom directory was provided (after & in the original list), use it.
    if [ -n "$custom_dir" ]; then
        repo_name="$custom_dir"
    else
        # Extract repo name from URL (handle both ssh and https forms)
        # Take substring after last '/' then strip optional .git suffix
        repo_name="${git_url##*/}"
        repo_name="${repo_name%.git}"
    fi

    repo_path="$IBRIDO_WS_SRC/$repo_name"

    if [ -d "$repo_path" ]; then
        echo "Pulling latest changes in $repo_name..."
        cd "$repo_path" || { echo "Failed to cd to $repo_path"; continue; }
        # run pull in background to parallelize; show which remote/branch being pulled if available
        (
            # optional: try to show current branch name
            current_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")"
            echo "  -> [$repo_name] branch: $current_branch"
            git pull
        ) &
        cd "$IBRIDO_WS_SRC" || { echo "Could not return to $IBRIDO_WS_SRC"; exit 1; }
    else
        echo "Directory $repo_path does not exist. Skipping..."
    fi
done

wait
echo 'update done.'

set -e