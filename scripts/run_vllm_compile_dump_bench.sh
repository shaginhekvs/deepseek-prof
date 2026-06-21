#!/usr/bin/env bash
set -euo pipefail

source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate

mkdir -p /scratch/deepseek-prof/profiles/vllm_compile_dump

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DEBUG_DUMP_PATH="/scratch/deepseek-prof/profiles/vllm_compile_dump"
export TORCH_LOGS="+dynamo,graph_breaks,recompiles,guards,aot_graphs,output_code"
export TORCH_LOGS_OUT="/scratch/deepseek-prof/profiles/vllm_compile_dump/torch_logs.txt"

vllm bench throughput \
  --model facebook/opt-125m \
  --dtype float16 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.35 \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 16 \
  --seed 0 \
  --output-json /scratch/deepseek-prof/profiles/bench/throughput_1gpu_vllm_compile_dump_opt125m_i512_o128_n16.json
