#!/usr/bin/env bash
set -euo pipefail

source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate

mkdir -p /scratch/deepseek-prof/profiles/ncu

# Nsight Compute is a per-kernel profiler. Keep this on a tiny workload first;
# full vLLM runs can generate very large reports and take much longer.
ncu \
  --target-processes all \
  --set speed-of-light \
  --force-overwrite \
  --export /scratch/deepseek-prof/profiles/ncu/torch_smoke_speed_of_light \
  --log-file /scratch/deepseek-prof/profiles/ncu/torch_smoke_speed_of_light.log \
  python /scratch/deepseek-prof/scripts/torch_cuda_kernel_smoke.py
