#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate sb_ov

# Returns the count of arguments that are in short or long options
VALID_ARGUMENTS=$# 

if [ "$VALID_ARGUMENTS" -eq 0 ]; then
  numactl -C 0 python stable_diffusion/bench_all.py
  exit 0
fi

# $1 can be --base_fp32 --base_bf16 --ipex_bf16 --ov_bf16
numactl -N 0 python stable_diffusion/bench_all.py $1