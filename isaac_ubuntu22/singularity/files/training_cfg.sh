#!/bin/bash

RB_PNAME="kyon"
RB_PPREFNAME="${HOME}/ibrido_ws/src/iit-kyon-ros-pkg"
SHM_NS="IBRIDO"
N_ENVS=128
RNAME="LinVelTrack"
SEED=0
ULIM_N=28672
TIMEOUT_MS=60000
CODEGEN_OVERRIDE_BDIR=""

SHM_NS+="_$(date '+%Y_%m_%d__%H_%M_%S')" # appending unique string to shm namespace
