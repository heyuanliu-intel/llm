#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate sb_ov

# Returns the count of arguments that are in short or long options
VALID_ARGUMENTS=$# 

if [ "$VALID_ARGUMENTS" -eq 0 ]; then
  numactl -C 0 python stable_diffusion/bench_all.py
  exit 0
fi

export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/sb_ov/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
# export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/pkgs/intel-openmp-2021.4.0-h06a4308_3561/lib/libiomp5.so
# export OMP_NUM_THREADS=112
# $1 can be --base_fp32 --base_bf16 --ipex_bf16 --ov_bf16
numactl -N 0 python stable_diffusion/bench_all.py $1