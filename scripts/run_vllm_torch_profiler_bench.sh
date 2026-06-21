#!/usr/bin/env bash
set -euo pipefail

source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate

mkdir -p /scratch/deepseek-prof/profiles/vllm_torch_profiler

export VLLM_ENABLE_V1_MULTIPROCESSING=0

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
  --profile \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/scratch/deepseek-prof/profiles/vllm_torch_profiler","torch_profiler_record_shapes":true,"torch_profiler_with_memory":true,"torch_profiler_with_stack":true,"torch_profiler_use_gzip":false,"warmup_iterations":1,"active_iterations":3,"ignore_frontend":true}' \
  --output-json /scratch/deepseek-prof/profiles/bench/throughput_1gpu_vllm_torch_profiler_opt125m_i512_o128_n16.json
