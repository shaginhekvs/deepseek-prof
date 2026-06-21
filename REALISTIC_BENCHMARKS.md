# Realistic Benchmark Notes

This phase moves from toy offline throughput to online serving behavior.

## Harness

Run the matrix:

```bash
NUM_PROMPTS=96 CONCURRENCIES="1 8 32" /scratch/deepseek-prof/scripts/run_serve_realistic_matrix.sh
```

Fast validation run:

```bash
NUM_PROMPTS=24 NUM_WARMUPS=2 CONCURRENCIES="1 8" /scratch/deepseek-prof/scripts/run_serve_realistic_matrix.sh
```

The matrix runs:

- `prefill_heavy`: random input around 1536 tokens, output around 32 tokens.
- `decode_heavy`: random input around 128 tokens, output around 512 tokens.
- `mixed_chat`: random input around 768 tokens, output around 256 tokens.

Results are saved under:

- `/scratch/deepseek-prof/profiles/serve_realistic/*.json`
- `/scratch/deepseek-prof/profiles/serve_realistic/summary.md`
- `/scratch/deepseek-prof/logs/serve_realistic/*.log`

Run the focused Nsight pass:

```bash
NUM_PROMPTS=24 CONCURRENCY=8 /scratch/deepseek-prof/scripts/run_nsys_serve_decode_heavy.sh
```

Artifacts from the first focused run:

- `/scratch/deepseek-prof/profiles/nsys/serve_decode_heavy_c8_ks-testdlaembeddinggen_105108.nsys-rep`
- `/scratch/deepseek-prof/profiles/nsys/serve_decode_heavy_c8_ks-testdlaembeddinggen_105108.stats.txt`
- `/scratch/deepseek-prof/profiles/serve_realistic/nsys_decode_heavy_c8_i128_o512_n24.json`

## First Results

Small validation run, `facebook/opt-125m`, 1 GPU, 24 prompts. This was collected before the harness defaulted to `TEMPERATURE=0`, so treat it as shape validation and first bottleneck signal, not the final deterministic baseline.

| workload | concurrency | req/s | out tok/s | mean TTFT ms | mean TPOT ms | mean ITL ms | mean E2E ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill_heavy | 1 | 19.72 | 610.56 | 15.20 | 1.18 | 1.18 | 50.62 |
| prefill_heavy | 8 | 90.16 | 2791.31 | 21.66 | 2.03 | 2.02 | 82.27 |
| decode_heavy | 1 | 1.65 | 811.46 | 7.01 | 1.22 | 1.22 | 604.95 |
| decode_heavy | 8 | 10.24 | 5026.80 | 11.20 | 1.43 | 1.43 | 710.33 |
| mixed_chat | 1 | 3.37 | 801.49 | 8.66 | 1.22 | 1.22 | 296.60 |
| mixed_chat | 8 | 17.28 | 4109.58 | 14.50 | 1.63 | 1.63 | 399.57 |

Reading:

- Concurrency increases throughput substantially.
- Concurrency also raises TTFT/TPOT/ITL, so latency SLOs decide the useful operating point.
- Decode-heavy is the best first optimization target because it dominates end-to-end latency and exposes steady token-step costs.
- Prefill-heavy throughput scales well, but TTFT climbs at concurrency 8; inspect queueing/scheduler behavior before optimizing kernels there.

## Nsight Decode-Heavy C8 Findings

From `/scratch/deepseek-prof/profiles/nsys/serve_decode_heavy_c8_ks-testdlaembeddinggen_105108.stats.txt`:

- FlashAttention split-KV: about 21.5% of GPU kernel time.
- Small-shape CUTLASS GEMM `cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_32x10_tn_align8`: about 21.1%.
- Signed-char fill kernel: about 7.9%.
- Other GEMMs, softmax/argmax, layernorm/relu Triton kernels, KV-cache reshape, and sampling kernels fill the rest.
- CUDA API time is dominated by kernel launches, `cudaMemGetInfo`, CUDA graph instantiation, memcpy, and graph launches.
- H2D memcpy dominates GPU mem-op time, but its absolute time is much smaller than kernel time.

Startup under Nsight was heavily distorted:

- Engine init took 79.43 s under full process-tree tracing.
- Compile cache load itself was only 1.37 s.
- CUDA graph profiling/capture under Nsight accounts for most of that startup distortion.

## Optimization Hypotheses

1. Decode attention path:
   - The FlashAttention split-KV kernel is the largest single family in decode-heavy serving.
   - Next profiler: `ncu` against the flash attention kernel family for achieved occupancy, memory throughput, and tensor-core utilization.

2. Small GEMM shape path:
   - A 64x64 CUTLASS GEMM family is nearly tied with attention.
   - For DeepSeek support, this maps conceptually to the “small batch/decode GEMM” problem: kernel selection and fusion matter more than big prefill GEMM throughput.

3. Metadata/fill overhead:
   - The signed-char fill kernel is too visible for a small model.
   - Look for mask/slot/block-table initialization or scheduler metadata clears happening every step.

4. Warmup gap:
   - Logs show `_compute_slot_mapping_kernel` Triton JIT during inference.
   - Extend warmup or explicitly cover the shape/config before measuring SLOs.

5. Launch overhead:
   - `cudaLaunchKernel` is the top CUDA API entry by total time under Nsight.
   - CUDA graphs help, but remaining uncaptured kernels and small ops are still relevant.

## Next Experiments

1. Repeat the matrix with `CONCURRENCIES="1 4 8 16 32"` and `NUM_PROMPTS=96`.
2. Compare `TEMPERATURE=0` against default sampling if sampler behavior is the suspected bottleneck.
3. Run a decode-heavy C8/C16 Nsight trace with a shorter captured window after warmup.
4. Use `ncu` on the top FlashAttention and 64x64 GEMM kernels.
5. For DeepSeek-V4-specific work, run the same matrix on the smallest viable DeepSeek-V4-compatible checkpoint or local synthetic module that exercises:
   - `vllm::deepseek_v4_attention`
   - MLA KV update/cache layout
   - MoE/top-k routing path
   - MTP speculative decode path
