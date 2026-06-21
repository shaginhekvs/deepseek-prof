#!/usr/bin/env bash
set -euo pipefail

source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate

mkdir -p /scratch/deepseek-prof/profiles/bench /scratch/deepseek-prof/logs

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,GRAPH,ENV,COLL}"
export NCCL_DEBUG_FILE="/scratch/deepseek-prof/logs/bench_tp4_nccl_%h_%p.log"
export NCCL_TOPO_DUMP_FILE="/scratch/deepseek-prof/logs/bench_tp4_nccl_topo_%h.xml"

vllm bench throughput \
  --model facebook/opt-125m \
  --dtype float16 \
  --max-model-len 1024 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.25 \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 64 \
  --seed 0 \
  --output-json /scratch/deepseek-prof/profiles/bench/throughput_tp4_opt125m_i512_o128_n64.json
