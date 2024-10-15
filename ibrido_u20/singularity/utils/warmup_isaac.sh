#!/bin/bash
ISAAC_ROOT="/isaac-sim"

set +e # Workaround post-install script failure

"$ISAAC_ROOT/python.sh" "/root/ibrido_utils/warmup_isaac.py"

