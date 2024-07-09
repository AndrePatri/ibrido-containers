#!/bin/bash
ISAAC_ROOT="/isaac-sim"

set +e # Workaround post-install script failure

"$ISAAC_ROOT/python.sh" "/usr/local/bin/warmup_isaac.py"
