#!/usr/bin/env bash
set -euo pipefail

source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,GRAPH,ENV,COLL}"
export NCCL_DEBUG_FILE="/scratch/deepseek-prof/logs/nccl_%h_%p.log"
export NCCL_TOPO_DUMP_FILE="/scratch/deepseek-prof/logs/nccl_topo_%h.xml"

torchrun --standalone --nproc-per-node=4 /scratch/deepseek-prof/scripts/nccl_allreduce_smoke.py
