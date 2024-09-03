#!/bin/bash
set +e # do not exit if any cmd fails

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_folder="$(dirname "$THIS_DIR")"

source "${root_folder}/files/bind_list.sh"

echo 'Updating workspace repos...'
cd $IBRIDO_WS_SRC
for ((i = 0; i < ${#IBRIDO_GITDIRS[@]}; i++)); do
    src="${IBRIDO_GIT_SRC[$i]}"
    echo "--> $src"
done
wait

echo 'Done.'

set -e
