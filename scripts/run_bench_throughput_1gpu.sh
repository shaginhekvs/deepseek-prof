#!/usr/bin/env bash
set -euo pipefail

source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate

mkdir -p /scratch/deepseek-prof/profiles/bench

vllm bench throughput \
  --model facebook/opt-125m \
  --dtype float16 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.35 \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 64 \
  --seed 0 \
  --output-json /scratch/deepseek-prof/profiles/bench/throughput_1gpu_opt125m_i512_o128_n64.json
