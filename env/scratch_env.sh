# Source this before building/running vLLM, PyTorch, profilers, or benchmarks.
# It keeps heavy source trees, caches, temp files, logs, and model artifacts off $HOME.

export DSPROF_ROOT=/scratch/deepseek-prof
export DSPROF_SRC="$DSPROF_ROOT/src"
export DSPROF_PROFILES="$DSPROF_ROOT/profiles"
export DSPROF_LOGS="$DSPROF_ROOT/logs"
export DSPROF_TMP="$DSPROF_ROOT/tmp"
export DSPROF_MODELS="$DSPROF_ROOT/models"
export DSPROF_DATASETS="$DSPROF_ROOT/datasets"
export DSPROF_ARTIFACTS="$DSPROF_ROOT/artifacts"

export TMPDIR="$DSPROF_TMP"
export TEMP="$DSPROF_TMP"
export TMP="$DSPROF_TMP"

export XDG_CACHE_HOME="$DSPROF_ROOT/cache"
export XDG_CONFIG_HOME="$DSPROF_ROOT/config"
export XDG_DATA_HOME="$DSPROF_ROOT/share"
export HF_HOME="$DSPROF_ROOT/cache/hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_ENABLE_HF_TRANSFER=1

export CUDA_CACHE_PATH="$DSPROF_ROOT/cache/nvidia/ComputeCache"
export TORCH_HOME="$DSPROF_ROOT/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$DSPROF_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$DSPROF_ROOT/cache/triton"
export PYTORCH_KERNEL_CACHE_PATH="$DSPROF_ROOT/cache/torch/kernels"

export PIP_CACHE_DIR="$DSPROF_ROOT/cache/pip"
export UV_CACHE_DIR="$DSPROF_ROOT/cache/uv"
export CCACHE_DIR="$DSPROF_ROOT/cache/ccache"
export SCCACHE_DIR="$DSPROF_ROOT/cache/sccache"
export RAY_TMPDIR="$DSPROF_ROOT/cache/ray"
export WANDB_DIR="$DSPROF_ROOT/cache/wandb"
export MPLCONFIGDIR="$DSPROF_ROOT/cache/matplotlib"

# Keep Python bytecode out of source trees when doing exploratory profiling.
export PYTHONPYCACHEPREFIX="$DSPROF_ROOT/cache/pycache"

# Useful defaults for traces and distributed debugging. Turn up only as needed.
export VLLM_LOGGING_LEVEL=INFO
# Leave TORCH_LOGS unset by default. Set it per run, for example:
#   export TORCH_LOGS="graph_breaks,recompiles,guards,dynamic"
unset TORCH_LOGS
export TORCHDYNAMO_VERBOSE=0
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,ENV
export NCCL_TOPO_DUMP_FILE="$DSPROF_LOGS/nccl_topo.xml"
export NCCL_DEBUG_FILE="$DSPROF_LOGS/nccl_%h_%p.log"

# Nsight Systems is on PATH at /usr/local/bin on this host. Avoid prepending
# /usr/local/cuda-13.2/bin because its nsys wrapper is not wired correctly here.
# Nsight Compute is installed but not always on PATH, so add it when present.
if [ -d /opt/nvidia/nsight-compute/2026.1.1 ]; then
  export PATH="/opt/nvidia/nsight-compute/2026.1.1:$PATH"
fi

mkdir -p \
  "$DSPROF_SRC" "$DSPROF_PROFILES" "$DSPROF_LOGS" "$DSPROF_TMP" \
  "$DSPROF_MODELS" "$DSPROF_DATASETS" "$DSPROF_ARTIFACTS" \
  "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" \
  "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" \
  "$CUDA_CACHE_PATH" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$PYTORCH_KERNEL_CACHE_PATH" \
  "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$CCACHE_DIR" "$SCCACHE_DIR" \
  "$RAY_TMPDIR" "$WANDB_DIR" "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX"
