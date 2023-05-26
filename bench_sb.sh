#!/bin/bash
export PATH="/root/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate sb_ov

# export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/sb_ov/lib/libjemalloc.so
# export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
# export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/pkgs/intel-openmp-2021.4.0-h06a4308_3561/lib/libiomp5.so
# export OMP_NUM_THREADS=112

export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libiomp5.so
export OMP_NUM_THREADS=112

# $1 can be --base_fp32 --base_bf16 --ipex_bf16 --ov_bf16
numactl -N 1 python stable_diffusion/bench.py $1