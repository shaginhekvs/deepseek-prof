#!/usr/bin/env bash
set -euo pipefail

source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate

mkdir -p /scratch/deepseek-prof/profiles/nsys /scratch/deepseek-prof/logs

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,GRAPH,ENV,COLL}"
export NCCL_DEBUG_FILE="/scratch/deepseek-prof/logs/nsys_nccl_%h_%p.log"
export NCCL_TOPO_DUMP_FILE="/scratch/deepseek-prof/logs/nsys_nccl_topo_%h.xml"

nsys profile \
  --output "/scratch/deepseek-prof/profiles/nsys/nccl_single_node_%h_%p" \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,nccl \
  --cuda-trace-scope=process-tree \
  --trace-fork-before-exec=true \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --wait=all \
  torchrun --standalone --nproc-per-node=4 /scratch/deepseek-prof/scripts/nccl_allreduce_smoke.py
