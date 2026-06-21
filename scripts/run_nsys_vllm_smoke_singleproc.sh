#!/usr/bin/env bash
set -euo pipefail

source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate

mkdir -p /scratch/deepseek-prof/profiles/nsys

export VLLM_ENABLE_V1_MULTIPROCESSING=0

nsys profile \
  --output "/scratch/deepseek-prof/profiles/nsys/vllm_smoke_singleproc_%h_%p" \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,nccl \
  --cuda-trace-scope=process-tree \
  --cuda-memory-usage=true \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  python /scratch/deepseek-prof/scripts/vllm_smoke.py
