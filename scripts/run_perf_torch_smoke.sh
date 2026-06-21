#!/usr/bin/env bash
set -euo pipefail

source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate

mkdir -p /scratch/deepseek-prof/profiles/cpu

perf record \
  -F 99 \
  -g \
  -o /scratch/deepseek-prof/profiles/cpu/torch_smoke_perf.data \
  -- python /scratch/deepseek-prof/scripts/torch_smoke_profile.py

perf report \
  -i /scratch/deepseek-prof/profiles/cpu/torch_smoke_perf.data \
  --stdio \
  --sort comm,dso,symbol \
  > /scratch/deepseek-prof/profiles/cpu/torch_smoke_perf_report.txt
