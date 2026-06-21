# Profiler Lab

All commands assume:

```bash
source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate
```

## What To Open Now

Start with these artifacts in this order:

1. `/scratch/deepseek-prof/profiles/torch_dynamo_inductor/dynamo_explain.txt`
2. `/scratch/deepseek-prof/profiles/torch_dynamo_inductor/exported_aten_graph.py`
3. `/scratch/deepseek-prof/cache/torchinductor_lab/nu/cnuxwatr56tf6kk3dnwmjtkmz4otj7drqvocs3lpolbgmemyffgs.debug/ir_pre_fusion.txt`
4. `/scratch/deepseek-prof/cache/torchinductor_lab/nu/cnuxwatr56tf6kk3dnwmjtkmz4otj7drqvocs3lpolbgmemyffgs.debug/ir_post_fusion.txt`
5. `/scratch/deepseek-prof/cache/torchinductor_lab/nu/cnuxwatr56tf6kk3dnwmjtkmz4otj7drqvocs3lpolbgmemyffgs.debug/output_code.py`
6. `/scratch/deepseek-prof/profiles/torch_dynamo_inductor/torch_profiler_table.txt`
7. `/scratch/deepseek-prof/profiles/vllm_torch_profiler/profiler_out_0.txt`
8. `/scratch/deepseek-prof/profiles/vllm_compile_dump/rank_0_dp_0/__compiled_fn_-1.before_split.0.py`
9. `/scratch/deepseek-prof/profiles/vllm_compile_dump/rank_0_dp_0/__compiled_fn_-1.after_split.0.py`
10. `/scratch/deepseek-prof/profiles/vllm_compile_dump/rank_0_dp_0/__compiled_fn_-1.kernel_0.py`
11. `/scratch/deepseek-prof/profiles/nsys/bench_1gpu_i512_o128_n64_ks-testdlaembeddinggen_88314.stats.txt`
12. `/scratch/deepseek-prof/profiles/nsys/nccl_single_node_ks-testdlaembeddinggen_83427.nsys-rep`
13. `/scratch/deepseek-prof/profiles/cpu/torch_smoke_perf_report.txt`
14. `/scratch/deepseek-prof/profiles/memory/vllm_smoke_memray.html`

Chrome trace files can be opened in Perfetto:

- `/scratch/deepseek-prof/profiles/torch_dynamo_inductor/torch_profiler_compile_lab.json`
- `/scratch/deepseek-prof/profiles/vllm_torch_profiler/rank0.1782032078946551118.pt.trace.json`
- `/scratch/deepseek-prof/profiles/torch/smoke/trace.json`

Nsight Systems traces can be opened with `nsys-ui` on a workstation, or summarized with:

```bash
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum /scratch/deepseek-prof/profiles/nsys/<trace>.nsys-rep
```

Nsight Compute CLI is installed at `/usr/local/cuda-13.2/bin/ncu` and is now added by `env/scratch_env.sh`. Reports and logs should be written explicitly under `/scratch/deepseek-prof/profiles/ncu`.

Scratch-backed Nsight paths:

- `TMPDIR=/scratch/deepseek-prof/tmp`
- `XDG_CACHE_HOME=/scratch/deepseek-prof/cache`
- `XDG_CONFIG_HOME=/scratch/deepseek-prof/config`
- `XDG_DATA_HOME=/scratch/deepseek-prof/share`
- `CUDA_CACHE_PATH=/scratch/deepseek-prof/cache/nvidia/ComputeCache`
- `/home/opc/.nv -> /scratch/deepseek-prof/cache/nvidia/home_nv`

Nsight Compute smoke command:

```bash
/scratch/deepseek-prof/scripts/run_ncu_torch_smoke.sh
```

Expected artifacts:

- `/scratch/deepseek-prof/profiles/ncu/torch_smoke_speed_of_light.ncu-rep`
- `/scratch/deepseek-prof/profiles/ncu/torch_smoke_speed_of_light.log`

Current local status: this wrapper successfully generated `/scratch/deepseek-prof/profiles/ncu/torch_smoke_speed_of_light.ncu-rep`. The log shows captured random-fill, GEMM, elementwise, copy, and reduction kernels. A Python `LookupError: unknown encoding: utf-8-sig` may appear after `ncu` exits or imports the report in the activated venv; the `.ncu-rep` is still valid and importable.

If it fails with `ERR_NVGPUCTRPERM`, GPU performance counters are restricted by the driver/admin policy. Nsight Systems will still work, but Nsight Compute hardware counters need that permission opened.

Do not run `ncu` around a script that also uses `torch.profiler`; both use CUPTI and can conflict. The smoke wrapper uses `torch_cuda_kernel_smoke.py`, which launches CUDA kernels without starting PyTorch's profiler.

## 1. PyTorch Profiler

Command:

```bash
python /scratch/deepseek-prof/scripts/torch_smoke_profile.py
```

Artifacts:

- `/scratch/deepseek-prof/profiles/torch/smoke/trace.json`

Use it for:

- CPU/CUDA op timeline.
- Kernel names associated with PyTorch ops.
- CUDA memory attribution when `profile_memory=True`.

## 2. Dynamo, Export, AOTAutograd, Inductor, Triton

Command:

```bash
/scratch/deepseek-prof/scripts/run_torch_dynamo_inductor_lab.sh
```

Artifacts:

- `/scratch/deepseek-prof/profiles/torch_dynamo_inductor/exported_aten_graph.py`
- `/scratch/deepseek-prof/profiles/torch_dynamo_inductor/dynamo_explain.txt`
- `/scratch/deepseek-prof/profiles/torch_dynamo_inductor/torch_logs.txt`
- `/scratch/deepseek-prof/profiles/torch_dynamo_inductor/torch_profiler_compile_lab.json`
- `/scratch/deepseek-prof/cache/torchinductor_lab`
- `/scratch/deepseek-prof/cache/triton_lab`

Use it for:

- Dynamo graph breaks, guards, recompiles.
- ATen-ish exported graph inspection.
- AOTAutograd graphs.
- Inductor output code and Triton kernel code.
- Mapping Python module structure to generated kernels.

## 3. vLLM Native PyTorch Profiler

Command:

```bash
/scratch/deepseek-prof/scripts/run_vllm_torch_profiler_bench.sh
```

Artifacts:

- `/scratch/deepseek-prof/profiles/vllm_torch_profiler`

Use it for:

- vLLM engine/worker PyTorch traces.
- Operator shapes and memory during a real vLLM workload.
- Compare against standalone `torch.profiler`.

## 4. vLLM Compile Dumps

Command:

```bash
/scratch/deepseek-prof/scripts/run_vllm_compile_dump_bench.sh
```

Artifacts:

- `/scratch/deepseek-prof/profiles/vllm_compile_dump`
- `/scratch/deepseek-prof/profiles/vllm_compile_dump/torch_logs.txt`
- `/scratch/deepseek-prof/cache/vllm/torch_compile_cache`

Use it for:

- vLLM FX/compile dump path.
- Dynamo and Inductor logs from vLLM's compiled model path.
- Cache hit/miss behavior across repeated runs.

## 5. Nsight Systems

Commands:

```bash
/scratch/deepseek-prof/scripts/run_nsys_vllm_smoke_singleproc.sh
/scratch/deepseek-prof/scripts/run_nsys_bench_throughput_1gpu_singleproc.sh
/scratch/deepseek-prof/scripts/run_nsys_nccl_single_node.sh
```

Artifacts:

- `/scratch/deepseek-prof/profiles/nsys/*.nsys-rep`
- `/scratch/deepseek-prof/profiles/nsys/*.sqlite`
- `/scratch/deepseek-prof/profiles/nsys/*.stats.txt`

Use it for:

- Whole-process CPU/CUDA/NCCL timeline.
- Kernel mix, CUDA API overhead, memcpys, CUDA graph capture/launch.
- NCCL collective kernels and synchronization.

Useful summaries:

```bash
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum /scratch/deepseek-prof/profiles/nsys/<file>.nsys-rep
```

## 6. NCCL

Commands:

```bash
/scratch/deepseek-prof/scripts/run_nccl_single_node.sh
/scratch/deepseek-prof/scripts/run_nsys_nccl_single_node.sh
/scratch/deepseek-prof/scripts/run_vllm_tp4_smoke.sh
```

Artifacts:

- `/scratch/deepseek-prof/logs/*nccl*.log`
- `/scratch/deepseek-prof/logs/*topo*.xml`
- `/scratch/deepseek-prof/profiles/nsys/nccl_single_node_*.nsys-rep`

Use it for:

- Local 4-GPU collective correctness.
- NCCL algorithm/protocol kernel visibility.
- Comparing synthetic collectives with vLLM TP=4 behavior.

## 7. CPU Profiling

Command:

```bash
/scratch/deepseek-prof/scripts/run_perf_torch_smoke.sh
```

Artifacts:

- `/scratch/deepseek-prof/profiles/cpu/torch_smoke_perf.data`
- `/scratch/deepseek-prof/profiles/cpu/torch_smoke_perf_report.txt`

Use it for:

- Native CPU hot symbols.
- Runtime overhead outside CUDA kernels.
- Threading/loader/scheduler overhead.

## 8. Python And CPU Memory

Command:

```bash
/scratch/deepseek-prof/scripts/run_memray_vllm_smoke.sh
```

Artifacts:

- `/scratch/deepseek-prof/profiles/memory/vllm_smoke_memray.bin`
- `/scratch/deepseek-prof/profiles/memory/vllm_smoke_memray.html`

Use it for:

- Python/native heap allocation flamegraph.
- CPU memory growth during vLLM model init and request processing.

## Reading Order

1. Open the PyTorch profiler trace to learn operator timeline basics.
2. Read `dynamo_explain.txt` and `exported_aten_graph.py`.
3. Read `torch_logs.txt` for graph breaks, guards, generated code.
4. Open the vLLM torch profiler trace.
5. Open Nsight Systems for the same workload.
6. Compare Nsight's kernel summary against PyTorch profiler's op summary.
7. Compare synthetic NCCL all-reduce to vLLM TP=4.
8. Use `perf`/`memray` only after GPU-side bottlenecks are understood.
