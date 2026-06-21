#!/usr/bin/env bash
set -euo pipefail

source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate

mkdir -p /scratch/deepseek-prof/profiles/memory

memray run \
  --force \
  -o /scratch/deepseek-prof/profiles/memory/vllm_smoke_memray.bin \
  /scratch/deepseek-prof/scripts/vllm_smoke.py

memray flamegraph \
  -o /scratch/deepseek-prof/profiles/memory/vllm_smoke_memray.html \
  /scratch/deepseek-prof/profiles/memory/vllm_smoke_memray.bin
