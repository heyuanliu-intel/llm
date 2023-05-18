#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate sb

# $1 can be --baseline --ipex
numactl -C 0-55 python stable_diffusion/bench_ov.py