#!/usr/bin/env bash
set -euo pipefail

source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate

mkdir -p /scratch/deepseek-prof/profiles/nsys

nsys profile \
  --output "/scratch/deepseek-prof/profiles/nsys/vllm_smoke_%h_%p" \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,nccl \
  --cuda-trace-scope=process-tree \
  --trace-fork-before-exec=true \
  --cuda-memory-usage=true \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --wait=all \
  python /scratch/deepseek-prof/scripts/vllm_smoke.py
