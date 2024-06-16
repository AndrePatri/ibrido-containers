#!/bin/bash
set -e # exiting if any cmd fails

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_folder="$(dirname "$THIS_DIR")"

source "${root_folder}/files/bind_list.sh"

echo 'Creating all directories...'
mkdir -p $IBRIDO_WS_SRC # to hold git repos
for item in "${IBRIDO_BDIRS_SRC[@]}"; do
    echo "--> $item"
    mkdir -p $item
done
echo 'Done.'

echo 'Creating all files...'
for item in "${IBRIDO_BFILES_SRC[@]}"; do
    echo "--> $item"
    dir_path=$(dirname "$item")
    mkdir -p $dir_path # create if doesn't exist
    touch $item
done
echo 'Done.'

echo 'Cloning repos...'
cd $IBRIDO_WS_SRC
for ((i = 0; i < ${#IBRIDO_GITDIRS[@]}; i++)); do
    src="${IBRIDO_GIT_SRC[$i]}"
    branch="${IBRIDO_GIT_BRCH[$i]}"
    echo "--> $src # $branch"
    git clone -q -b $branch $src &
done
wait

# copying some utility files
cp ${root_folder}/files/setup.bash $IBRIDO_WS_PREFIX/
cp ${root_folder}/files/mamba_env.yml $IBRIDO_CONDA/

echo 'Done.'
