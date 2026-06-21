#!/usr/bin/env bash
set -euo pipefail

source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate

mkdir -p /scratch/deepseek-prof/profiles/torch_dynamo_inductor

export TORCH_LOGS="+dynamo,graph_breaks,recompiles,guards,aot_graphs,output_code,kernel_code"
export TORCH_LOGS_OUT="/scratch/deepseek-prof/profiles/torch_dynamo_inductor/torch_logs.txt"
export TORCH_COMPILE_DEBUG=1
export TORCHINDUCTOR_CACHE_DIR="/scratch/deepseek-prof/cache/torchinductor_lab"
export TRITON_CACHE_DIR="/scratch/deepseek-prof/cache/triton_lab"

python /scratch/deepseek-prof/scripts/torch_dynamo_inductor_lab.py
