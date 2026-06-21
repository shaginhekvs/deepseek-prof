# DeepSeek vLLM Profiling and Kernel Optimization Lab

This repository is a hands-on inference-engineering lab for understanding the path from PyTorch model code to GPU kernels, then using that understanding to profile vLLM/DeepSeek-style inference and search for real optimizations.

The main thread:

1. Follow the PyTorch execution path from eager code into Dynamo, ATen IR, Inductor IR, Triton/CUDA kernels, and profiler traces.
2. Run vLLM inference workloads under PyTorch Profiler, Nsight Systems, Nsight Compute, CPU profilers, memory profilers, and NCCL/network diagnostics.
3. Inspect the bottlenecks in realistic traces instead of optimizing random kernels.
4. Try open-source kernel libraries and DSLs, including Helion, HF kernels, CUTLASS/CuTeDSL, Liger-style kernels, and vLLM's own custom ops.
5. Identify a real potential optimization in the DeepSeek vLLM pipeline: fusing DeepSeek's FP16 scale step with residual RMSNorm using Helion, giving about a 3x microbenchmark speedup on A10 for larger DeepSeek-hidden shapes.

## Environment

The lab is designed to keep caches, traces, build outputs, and model artifacts on `/scratch`, not under `~`.

Use this before running experiments:

```bash
source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate
cd /scratch/deepseek-prof
```

Important local paths:

- vLLM checkout: `/scratch/deepseek-prof/src/vllm`
- PyTorch/vLLM scratch caches: `/scratch/deepseek-prof/cache`
- profiler outputs: `/scratch/deepseek-prof/profiles`
- benchmark scripts: `/scratch/deepseek-prof/scripts`

## Profiling Path

The profiling workflow starts with small, inspectable PyTorch programs and then moves into vLLM serving and benchmark workloads.

Core references:

- [RUNBOOK.md](RUNBOOK.md): setup, cache redirection, source checkouts, profiler commands.
- [PROFILER_LAB.md](PROFILER_LAB.md): PyTorch Profiler, Dynamo/Inductor/ATen inspection, Nsight Systems, Nsight Compute, CPU, memory, and NCCL notes.
- [REALISTIC_BENCHMARKS.md](REALISTIC_BENCHMARKS.md): realistic vLLM workload matrix and trace-reading workflow.
- [MEDIUM_ARTICLE.md](MEDIUM_ARTICLE.md): narrative article draft for the whole investigation.

Article diagrams:

- [diagrams/pytorch_to_gpu_stack.svg](diagrams/pytorch_to_gpu_stack.svg)
- [diagrams/profiling_optimization_loop.svg](diagrams/profiling_optimization_loop.svg)
- [diagrams/deepseek_helion_fusion.svg](diagrams/deepseek_helion_fusion.svg)
- [diagrams/a10_gpu_topology.svg](diagrams/a10_gpu_topology.svg)

The mental model is:

```text
PyTorch model code
  -> torch.compile / Dynamo graph capture
  -> ATen graph / FX graph inspection
  -> Inductor IR and generated code
  -> Triton/CUDA/vLLM custom kernels
  -> GPU timeline, kernel stats, CPU runtime, memory, NCCL
```

For vLLM/DeepSeek, this means looking at both Python model code and the lower-level operators it dispatches to. A useful example is `RMSNorm`: DeepSeek model code calls vLLM's `RMSNorm`, which routes residual cases through `ir.ops.fused_add_rms_norm`, which can dispatch to native PyTorch, vLLM C kernels, or an experimental Helion provider.

## Kernel Search

After profiling, the lab compares several kernel sources:

- vLLM C/custom ops
- Helion kernels and autotuned Helion configs
- Hugging Face `kernels-community/activation`
- CUTLASS and CuTeDSL experiments
- Liger/quack-style normalization and activation kernels where available

References:

- [KERNEL_CANDIDATES.md](KERNEL_CANDIDATES.md): candidate libraries and where each might fit.
- [HELION_EXPERIMENTS.md](HELION_EXPERIMENTS.md): initial Helion activation/GEMM/fusion experiments.
- [HELION_A10_FUSION_FINDINGS.md](HELION_A10_FUSION_FINDINGS.md): A10-specific Helion fusion findings.
- [CUTLASS_EXPERIMENTS.md](CUTLASS_EXPERIMENTS.md): CUTLASS/CuTeDSL comparison notes.

The important rule is that a kernel only matters if it maps back to an operation in the vLLM DeepSeek pipeline.

## DeepSeek vLLM Findings

### 0. A10 GPU Topology

This server has four local A10 GPUs, but no NVLink/NVSwitch fabric. `nvidia-smi topo -m` shows two closer PCIe/NUMA GPU islands:

```text
GPU0 <-> GPU1: NODE
GPU2 <-> GPU3: NODE
cross-island pairs such as GPU0 <-> GPU2: SYS
```

NVLS is therefore not applicable on this host. The practical question is how much the cross-NUMA `SYS` path costs.

Benchmark:

- [scripts/bench_gpu_topology_pairs.py](scripts/bench_gpu_topology_pairs.py)
- [profiles/gpu_topology/gpu_topology_pairs.md](profiles/gpu_topology/gpu_topology_pairs.md)

Measured `0,1` versus `0,2`:

```text
CUDA peer copy, 256MB:
0 -> 1: 21.96 GB/s
0 -> 2: 18.55 GB/s
loss: about 15.5%

NCCL 2-GPU all-reduce, 256MB:
0,1: 15.24 GB/s per rank
0,2: 12.79 GB/s per rank
loss: about 16.1%
```

The mirrored pair shows the same pattern:

```text
2 -> 3 peer copy: 22.02 GB/s
1 -> 3 peer copy: 18.53 GB/s

2,3 NCCL 256MB: 15.14 GB/s per rank
1,3 NCCL 256MB: 12.94 GB/s per rank
```

Practical placement rule: prefer tightly-coupled tensor-parallel or frequent MoE communication inside `0,1` or inside `2,3`. Crossing between the two islands is still usable, but large transfers and collectives cost roughly `1.19x` time on this machine.

### 1. `fused_add_rms_norm`

DeepSeek decoder layers use residual RMSNorm through vLLM:

```text
DeepseekV2DecoderLayer
  -> RMSNorm(..., residual=...)
  -> ir.ops.fused_add_rms_norm.maybe_inplace(...)
```

An experimental Helion provider was added locally in the vLLM checkout:

- `/scratch/deepseek-prof/src/vllm/vllm/kernels/helion_ops.py`
- enabled with `VLLM_USE_HELION_OPS=1`

Benchmark:

- [scripts/bench_vllm_deepseek_helion_rmsnorm.py](scripts/bench_vllm_deepseek_helion_rmsnorm.py)
- [profiles/helion_vllm_ir/vllm_deepseek_helion_rmsnorm.md](profiles/helion_vllm_ir/vllm_deepseek_helion_rmsnorm.md)
- [DEEPSEEK_HELION_INTEGRATION.md](DEEPSEEK_HELION_INTEGRATION.md)

Result: Helion is competitive with vLLM C for larger hidden-size residual RMSNorm shapes, but vLLM C still dominates small decode/MLA-rank shapes.

### 2. DeepSeek FP16 Scale Plus Residual RMSNorm

The strongest finding is in the DeepSeek MLA FP16 overflow path. DeepSeek scales `hidden_states` and `residual` before post-attention RMSNorm:

```text
hidden_states *= scale
residual *= scale
hidden_states, residual = post_attention_layernorm(hidden_states, residual)
```

The experiment fuses this into one Helion kernel:

```text
y = (hidden_states + residual) * scale
residual_out = y
hidden_states_out = rms_norm(y, weight, eps)
```

Benchmark:

- [scripts/bench_deepseek_pipeline_fusions.py](scripts/bench_deepseek_pipeline_fusions.py)
- [profiles/deepseek_fusions/deepseek_pipeline_fusions.md](profiles/deepseek_fusions/deepseek_pipeline_fusions.md)
- [DEEPSEEK_FUSION_EXPERIMENTS.md](DEEPSEEK_FUSION_EXPERIMENTS.md)

Current A10 result:

```text
1024x3072: scale+vLLM C 0.1637 ms, Helion fused 0.0531 ms, 3.09x
8192x3072: scale+vLLM C 1.2563 ms, Helion fused 0.4057 ms, 3.10x
```

This is the best candidate found so far for a real DeepSeek/vLLM path optimization.

### 3. DeepSeek MLP `SiluAndMul`

DeepSeek dense MLP uses:

```text
gate_up_proj -> SiluAndMul -> down_proj
```

Results:

- HF activation kernel wins on one small shape (`128x2048`).
- vLLM C is strong on common shapes.
- Helion only slightly beats vLLM C on larger activation shapes.

Decision: do not replace globally. A small-batch, env-gated HF activation experiment may be worth trying, but the scale-plus-RMSNorm fusion is a better target.

### 4. CuTeDSL/CUTLASS MoE

CUTLASS DSL is installed, but vLLM's FlashInfer CuTeDSL MoE paths in this checkout are mainly SM100/NVFP4-oriented. This host is A10 (`sm_86`), so those paths are not a direct A10 DeepSeek MoE win.

## Reproduce Key Result

```bash
source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate
cd /scratch/deepseek-prof
python scripts/bench_deepseek_pipeline_fusions.py --warmup 10 --iters 50
```

Open:

```text
/scratch/deepseek-prof/profiles/deepseek_fusions/deepseek_pipeline_fusions.md
```

## Next Steps

The next engineering step is to move the `scale + residual RMSNorm` Helion fusion from the benchmark harness into an env-gated vLLM DeepSeek path, then run a real serving benchmark and Nsight Systems trace.

Candidate integration path:

1. Add an env flag such as `VLLM_USE_HELION_DEEPSEEK_SCALE_RMSNORM=1`.
2. Patch only the DeepSeek FP16 MLA branch around `post_attention_layernorm`.
3. Guard by dtype, shape, CUDA device, and row count.
4. Compare end-to-end serving latency and throughput against baseline vLLM.
5. Use Nsight Systems and PyTorch Profiler to verify the separate scale kernels disappeared and the new Helion kernel appears in the right place.
