#!/bin/bash

cd "$(dirname "$0")"
mkdir ../build
singularity build ../build/cudagl2204.sif cudagl2204.def
