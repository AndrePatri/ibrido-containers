#!/bin/bash

sudo singularity registry login --username \$oauthtoken docker://nvcr.io
sudo singularity build ./ibrido_isaac.sif ./u22_isaac.def

