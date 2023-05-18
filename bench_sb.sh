#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate sb

# Returns the count of arguments that are in short or long options
VALID_ARGUMENTS=$# 

if [ "$VALID_ARGUMENTS" -eq 0 ]; then
  numactl -C 0 python stable_diffusion/bench.py
  exit 0
fi

# $1 can be --baseline --ipex
numactl -C 0 python stable_diffusion/bench.py $1