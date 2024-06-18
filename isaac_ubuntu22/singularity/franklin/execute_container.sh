#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source $SCRIPT_DIR/run_cfg.sh

$SCRIPT_DIR/../execute.sh --wandb_key $WANDB_KEY [--comment|-c <key>]"