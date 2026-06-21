# DeepSeek V4 vLLM/PyTorch Profiling Runbook

## Current Machine State

- Scratch workspace: `/scratch/deepseek-prof`
- Source trees:
  - vLLM: `/scratch/deepseek-prof/src/vllm`
  - PyTorch: `/scratch/deepseek-prof/src/pytorch`
  - DeepSeek V4 PR ref in vLLM: `pr-40760-deepseek-v4`
- Installed host tools: `git`, `python3-pip`, `nsys`, `perf`, `iperf3`, `numactl`, `numastat`, `ibstat`, `ib_send_bw`, `ib_write_bw`, `ib_read_bw`, `ib_read_lat`, `ib_write_lat`
- GPU topology: one server with 4 local NVIDIA A10 GPUs visible via `nvidia-smi`.
- CUDA base: PyTorch CUDA smoke tests pass on all 4 GPUs with `torch==2.11.0+cu128`.
- vLLM base: editable precompiled install works in `/scratch/deepseek-prof/env/py312`.
- CUDA toolkit caveat: root is too small for the full RPM CUDA toolkit; `nvcc` and `ncu` are not available as full native tools. Use precompiled vLLM for Python/model/scheduler work. Native CUDA kernel development needs a CUDA toolkit mounted or installed on `/scratch`, or a larger root volume.
- Local topology: GPU0/GPU1 are NUMA node 0 near `mlx5_0`; GPU2/GPU3 are NUMA node 1 near `mlx5_1`.

Always start shells with:

```bash
source /scratch/deepseek-prof/env/scratch_env.sh
```

## Scratch-First Storage Policy

The environment file redirects these heavy paths to `/scratch/deepseek-prof`:

- `TMPDIR`, `TMP`, `TEMP`
- `XDG_CACHE_HOME`
- `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`
- `TORCH_HOME`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `PYTORCH_KERNEL_CACHE_PATH`
- `PIP_CACHE_DIR`, `UV_CACHE_DIR`
- `CCACHE_DIR`, `SCCACHE_DIR`
- `RAY_TMPDIR`
- `WANDB_DIR`, `MPLCONFIGDIR`
- `PYTHONPYCACHEPREFIX`

Before any long run:

```bash
source /scratch/deepseek-prof/env/scratch_env.sh
env | sort | grep -E '^(TMP|XDG|HF|HUGGINGFACE|TRANSFORMERS|TORCH|TRITON|PIP|UV|RAY|WANDB|MPL|PYTHONPYCACHE)'
df -h /scratch /home/opc
```

## DeepSeek V4 vLLM Development Map

The PR is broad. Treat it as several systems, not one feature:

- Model semantics: `vllm/model_executor/models/deepseek_v4.py`, `deepseek_v4_mtp.py`, HF config conversion.
- Tokenization/rendering/tools: `vllm/tokenizers/deepseek_v4.py`, `vllm/renderers/deepseek_v4.py`, `vllm/tool_parsers/deepseekv4_tool_parser.py`.
- Attention and KV cache: `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/v1/attention/ops/deepseek_v4_ops/*`, sparse MLA/indexer code.
- MoE: fused MoE router/runner changes, `deep_gemm_mega_moe`, top-k softplus/sqrt kernels, expert parallel paths.
- Custom CUDA: `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`, compressor/cache kernels, top-k, sampler.
- Serving behavior: v1 scheduler, KV cache coordinator, GPU model runner, speculative MTP.

The checked-in eval recipe uses:

```bash
--trust-remote-code \
--kv-cache-dtype fp8 \
--block-size 256 \
--enable-expert-parallel \
--tensor-parallel-size 2 \
--attention_config.use_fp4_indexer_cache=True \
--moe-backend deep_gemm_mega_moe \
--tokenizer-mode deepseek_v4 \
--tool-call-parser deepseek_v4 \
--enable-auto-tool-choice \
--reasoning-parser deepseek_v4 \
--speculative_config.method=mtp \
--speculative_config.num_speculative_tokens=2
```

## Profilers To Use

Use these in this order:

1. `vllm bench` and service logs for workload-level throughput/latency, TTFT, ITL, request queueing, prefill/decode balance, and scheduler behavior.
2. `torch.profiler` for PyTorch op timelines, CPU/GPU correlation, memory attribution, shapes, and Chrome traces.
3. `TORCH_LOGS` plus `torch._dynamo.explain` for Dynamo graph breaks, guards, recompiles, and capture boundaries.
4. `TORCH_COMPILE_DEBUG=1`, `TORCHINDUCTOR_CACHE_DIR`, and Inductor debug dumps for FX graph, AOTAutograd/ATen graph, scheduler IR, generated Triton/C++/CUDA.
5. `nsys` for end-to-end CPU threads, CUDA API launches, kernels, NVTX ranges, NCCL collectives, OS runtime, and process-tree behavior.
6. `ncu` for individual hot CUDA kernels after `nsys` identifies them.
7. `NCCL_DEBUG`, `nccl-tests`, and `nsys --trace=nccl,cuda,nvtx,osrt` for collective topology, algorithm/protocol behavior, and overlap.
8. `torchrun` NCCL smoke tests, `NCCL_DEBUG`, `nvidia-smi topo -m`, and Nsight NCCL traces for local 4-GPU collectives. Use `ib_*`/`iperf3` only for NIC health or external serving traffic; there are no remote GPU nodes in this setup.
9. `perf`, `py-spy`, `memray`, `psutil`, `numastat`, `/proc/<pid>/smaps_rollup`, and cgroup files for CPU, Python, native stacks, RSS/PSS, NUMA locality, and allocator behavior.

## PyTorch Profiler Template

Use this around a minimal offline generation path first, then around vLLM worker/model-runner internals.

```python
import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

trace_dir = "/scratch/deepseek-prof/profiles/torch"

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=2, warmup=2, active=4, repeat=1),
    on_trace_ready=tensorboard_trace_handler(trace_dir),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_modules=True,
) as prof:
    for step in range(12):
        run_one_inference_step()
        prof.step()

print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=50))
```

Open traces:

```bash
tensorboard --logdir /scratch/deepseek-prof/profiles/torch --port 6006
```

## Dynamo, ATen, and Inductor Inspection

Start light:

```bash
export TORCH_LOGS="graph_breaks,recompiles,guards,dynamic"
export TORCHDYNAMO_VERBOSE=1
```

Turn on deep compiler dumps for a short reproduction only:

```bash
export TORCH_COMPILE_DEBUG=1
export TORCHINDUCTOR_CACHE_DIR=/scratch/deepseek-prof/cache/torchinductor
export TRITON_CACHE_DIR=/scratch/deepseek-prof/cache/triton
```

Useful code probes:

```python
import torch
import torch._dynamo as dynamo

print(dynamo.explain(fn)(*example_args))

exported = torch.export.export(module, example_args)
print(exported.graph_module.code)  # normalized ATen-ish graph
```

For Inductor output, inspect:

```bash
find /scratch/deepseek-prof/cache/torchinductor -type f | sort | head
find /scratch/deepseek-prof/cache/torchinductor -type f \( -name '*.py' -o -name '*.cu' -o -name '*.ttir' -o -name '*.ttgir' \)
```

## Nsight Systems

Use this for local vLLM profiling. For detailed CUDA kernel attribution, prefer the single-process wrapper because vLLM's default V1 multiprocessing can hide kernel summaries from a simple `nsys stats` pass:

```bash
/scratch/deepseek-prof/scripts/run_nsys_vllm_smoke_singleproc.sh
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  /scratch/deepseek-prof/profiles/nsys/vllm_smoke_singleproc_*.nsys-rep
```

For the default multi-process serving shape, keep process-tree tracing enabled:

```bash
source /scratch/deepseek-prof/env/scratch_env.sh
nsys profile \
  --output /scratch/deepseek-prof/profiles/nsys/vllm_%h_%p \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,nccl \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --cuda-memory-usage=true \
  --capture-range=nvtx \
  --capture-range-end=stop \
  python /scratch/deepseek-prof/scripts/vllm_smoke.py
```

For a first pass without NVTX capture control:

```bash
nsys profile \
  --duration 90 \
  --output /scratch/deepseek-prof/profiles/nsys/vllm_smoke_%h_%p \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,nccl \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  vllm serve deepseek-ai/DeepSeek-V4-Flash ...
```

Summaries:

```bash
nsys stats /scratch/deepseek-prof/profiles/nsys/*.nsys-rep
nsys export --type sqlite --output /scratch/deepseek-prof/profiles/nsys/export.sqlite /scratch/deepseek-prof/profiles/nsys/run.nsys-rep
```

## Nsight Compute

Use only after `nsys` names the hot kernels. Keep captures narrow.

```bash
ncu \
  --target-processes all \
  --kernel-name regex:fused_deepseek|deepgemm|topk|mla|compress \
  --set full \
  --export /scratch/deepseek-prof/profiles/ncu/hot_kernel \
  vllm serve deepseek-ai/DeepSeek-V4-Flash ...
```

## Local NCCL and Topology Profiling

Baseline local GPU topology:

```bash
nvidia-smi topo -m
numactl --hardware
```

Single-node NCCL smoke:

```bash
/scratch/deepseek-prof/scripts/run_nccl_single_node.sh
```

vLLM TP=4 smoke:

```bash
/scratch/deepseek-prof/scripts/run_vllm_tp4_smoke.sh
```

Useful environment:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,ENV,COLL
export NCCL_DEBUG_FILE=/scratch/deepseek-prof/logs/nccl_%h_%p.log
export NCCL_TOPO_DUMP_FILE=/scratch/deepseek-prof/logs/nccl_topo_%h.xml
```

Observed on this server:

- `torchrun --nproc-per-node=4` NCCL all-reduce passes.
- vLLM `tensor_parallel_size=4` passes.
- vLLM selects PYNCCL for tensor-parallel all-reduce.
- vLLM disables custom all-reduce on more than two PCIe-only GPUs.
- Symmetric memory communicator is not available on A10 compute capability 8.6.

If RoCE interface selection is ambiguous, test explicitly:

```bash
export NCCL_SOCKET_IFNAME=<interface>
export NCCL_IB_HCA=mlx5_0,mlx5_1
```

## CPU and CPU Memory

Live process view:

```bash
pidstat -rud -h -p <pid> 1
numastat -p <pid>
cat /proc/<pid>/smaps_rollup
cat /proc/<pid>/status | egrep 'VmRSS|VmHWM|Threads|voluntary|nonvoluntary'
```

CPU flame graph:

```bash
perf record -F 99 -g -p <pid> -- sleep 60
perf report
```

Python stacks:

```bash
python3 -m pip install --cache-dir /scratch/deepseek-prof/cache/pip py-spy memray
py-spy top --pid <pid>
py-spy record -o /scratch/deepseek-prof/profiles/cpu/pyspy.svg --pid <pid> --duration 60
memray run -o /scratch/deepseek-prof/profiles/memory/memray.bin your_script.py
memray flamegraph -o /scratch/deepseek-prof/profiles/memory/memray.html /scratch/deepseek-prof/profiles/memory/memray.bin
```

## Development Loop For Inference Engineering

1. Make the smallest reproducible workload: one prompt, fixed lengths, fixed batch/concurrency, deterministic sampling.
2. Validate model wiring before optimizing: tokenizer, config mapping, weight loading, attention masks, KV layout, MTP outputs, tool/reasoning parser.
3. Add correctness tests for each support boundary: config, tokenizer, parser, shape math, custom op numerical parity, scheduler/KV cache behavior.
4. Use eager/reference PyTorch or existing DeepSeek V3/V3.2 paths as the oracle.
5. Profile one GPU first, then TP=4 on the same server. Separate prefill, decode, MTP/spec decode, MoE routing, tensor-parallel all-reduce, and KV cache writes.
6. Compare app-level latency against local NCCL all-reduce and vLLM TP=4 smoke traces.
7. Move from timeline to kernel detail: `nsys` first, `ncu` only for kernels already proven hot.
8. Keep every trace and cache in `/scratch/deepseek-prof`; never allow model downloads, Inductor caches, Triton autotune caches, or Ray tmpdirs into home.

## Immediate Next Steps

1. Use single-process vLLM for local CUDA kernel attribution:
   `run_nsys_vllm_smoke_singleproc.sh`
2. Use default vLLM multiprocessing for realistic TP=4 behavior:
   `run_vllm_tp4_smoke.sh`
3. Use fixed offline throughput benchmarks:
   - 1 GPU: `run_bench_throughput_1gpu.sh`
   - TP=4: `run_bench_throughput_tp4.sh`
4. Add DeepSeek-family model smoke tests that fit A10 memory, then move toward DeepSeek V4 config/tokenizer/scheduler work.
5. For native DeepSeek V4 CUDA kernel development, provide a full CUDA toolkit on `/scratch` or a larger root volume so `nvcc` and `ncu` are available.

Current fixed-workload baseline with `facebook/opt-125m`, 64 requests, 511 input tokens, 128 output tokens:

- 1 GPU: 84.04 requests/s, 53,786.61 tokens/s.
- TP=4: 108.75 requests/s, 69,599.84 tokens/s.
