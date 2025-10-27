#!/usr/bin/env bash
set -eu

usage() {
cat > /dev/stderr << EOF
Usage: artifactory-tool.sh [[option1] ...] <operation>

Operations:
   set-profile <profile_name>    Setup all repositories listed in profile_name

Options:
   --perform-update              After modifying sources.list, perform apt-get update

Available <profile_name>s:
   default        Use repo settings for unapproved packages (urm cache)
   approved       The Omniverse Approved artifactory repositories for each of:
                    * apt/.deb (urm approved)
                    * pypi
                    * npm
   canonical      Use repo settings which ship by default with the upstream Ubuntu container (ubuntu.com)
EOF
}

APT_LIST_CANONICAL_PATH=/etc/apt/sources.list.canonical
APT_LIST_URMCACHE_PATH=/etc/apt/sources.list.urmcache
APT_LIST_URMAPPROVED_PATH=/etc/apt/sources.list.urmapproved
APT_LIST_PATH=/etc/apt/sources.list

if [[ $# -lt 2 ]]; then
    usage
    exit 1
fi

perform_update=0
if [[ "$1" == "--perform-update" ]]; then
    perform_update=1
    shift
fi

add_deb_src=0
if [[ "$1" == "--add-deb-src" ]]; then
    add_deb_src=1
    shift
fi

operation=$1
profile_name=$2

if [[ $operation != "set-profile" ]]; then
    echo -e "Unsupported operation: $operation\n" > /dev/stderr
    usage
    exit 1
fi

apt_mods_common() {
    if [[ "$add_deb_src" == 1 ]]; then
        grep '^deb ' /etc/apt/sources.list | sed -e 's/^deb /deb-src /' > /tmp/deb-src.list
        cat /tmp/deb-src.list >> /etc/apt/sources.list
    fi

    if [[ "$perform_update" == 1 ]]; then
        apt-get update
    fi
}

if [[ $profile_name = "default" ]]; then
    cp $APT_LIST_URMCACHE_PATH $APT_LIST_PATH
    if [[ "$perform_update" == 0 ]]; then
        echo "APT sources.list modified, remember to run 'apt-get update'"
    fi

    sed -i '/^index-url = https:[/][/]urm.nvidia/d' /root/.pip/pip.conf
    echo "index-url override removed from pip.conf"

    sed -i '/registry=https:[/][/]urm.nvidia/d' /root/.npmrc
    echo "registry override removed from .npmrc"

    apt_mods_common

elif [[ $profile_name = "canonical" ]]; then
    cp $APT_LIST_CANONICAL_PATH $APT_LIST_PATH
    if [[ "$perform_update" == 0 ]]; then
        echo "APT sources.list modified, remember to run 'apt-get update'"
    fi

    sed -i '/^index-url = https:[/][/]urm.nvidia/d' /root/.pip/pip.conf
    echo "index-url override removed from pip.conf"

    sed -i '/registry=https:[/][/]urm.nvidia/d' /root/.npmrc
    echo "registry override removed from .npmrc"

    apt_mods_common

elif [[ $profile_name = "approved" ]]; then
    cp $APT_LIST_URMAPPROVED_PATH $APT_LIST_PATH
    if [[ "$perform_update" == 0 ]]; then
        echo "apt sources.list modified to reference urm.nvidia.com only, remember to run 'apt-get update'"
    fi
    [[ -d /root/.pip ]] || mkdir -p /root/.pip
    echo -e '[global]\nindex-url = https://urm.nvidia.com/artifactory/api/pypi/ct-omniverse-mirror-pypi/simple\n' >> /root/.pip/pip.conf
    echo "index-url override added to pip.conf"

    echo 'registry=https://urm.nvidia.com/artifactory/api/npm/ct-omniverse-mirror-npm/' >> /root/.npmrc
    echo "registry override added to .npmrc"

    apt_mods_common

else
    echo -e "Unsupported profile name: $profile_name\n" > /dev/stderr
    usage
    exit 1

fi

