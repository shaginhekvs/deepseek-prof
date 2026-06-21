# Helion Kernel Experiments

Helion is worth trying in this project as a rapid kernel-authoring and autotuning layer. It is not a replacement for Nsight-driven work: use it when a trace has already identified a hot, shape-stable op and you want to generate/tune a candidate faster than writing raw Triton or CUDA by hand.

Sources:

- `https://github.com/pytorch/helion`
- `https://pytorch.org/blog/helion/`
- vLLM RFC: `https://github.com/vllm-project/vllm/issues/32219`

## Installed Package

Installed in the scratch venv:

```bash
source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate
python -c "import helion; print(helion.__file__)"
```

The first lab uses Helion `1.1.0`.

## Run

```bash
source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate
python /scratch/deepseek-prof/scripts/helion_kernel_lab.py --warmup 10 --iters 50 --shapes 1024x3072 8192x3072
```

Artifacts:

- `/scratch/deepseek-prof/profiles/helion/helion_kernel_lab.json`
- `/scratch/deepseek-prof/profiles/helion/helion_kernel_lab.md`
- `/scratch/deepseek-prof/profiles/helion/silu_and_mul_autotune.csv`
- `/scratch/deepseek-prof/profiles/helion/silu_and_mul_autotune.log`

The generated Triton/Inductor code paths are printed during the run and live under `/scratch/deepseek-prof/cache/torchinductor/...`.

## Current Result

On the local A10, fused `silu_and_mul`:

- `(1024, 3072)`: Torch `0.0759 ms`, Helion `0.0410 ms`, about `1.85x` faster.
- `(8192, 3072)`: Torch `0.5382 ms`, Helion `0.3092 ms`, about `1.74x` faster.
- HF `kernels-community/activation` is very close: `0.0420 ms` and `0.3131 ms`.

This means Helion is already competitive with a purpose-built fused activation kernel for this simple op.

## GEMM Bottleneck Probe

I also tested a plain Helion GEMM against the small decode-linear proxy shapes from the current vLLM bottleneck investigation.

Run:

```bash
source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate
python /scratch/deepseek-prof/scripts/helion_gemm_bottleneck_lab.py --warmup 10 --iters 50
```

Artifacts:

- `/scratch/deepseek-prof/profiles/helion_gemm/helion_gemm_bottleneck_lab.json`
- `/scratch/deepseek-prof/profiles/helion_gemm/helion_gemm_bottleneck_lab.md`

Result: plain Helion GEMM did not beat PyTorch/cuBLAS for the proxy decode shapes.

- `M=8/16/32, K=768, N=3072`: Helion was about `0.29-0.35x` of `torch.mm`.
- `M=8/16/32, K=3072, N=768`: Helion was about `0.37-0.42x` of `torch.mm`.
- `M=8/16/32, K=768, N=768`: Helion was about `0.41-0.43x` of `torch.mm`.

Interpretation: do not integrate a plain Helion GEMM into vLLM for this bottleneck. The better Helion path remains fused elementwise kernels or GEMM-adjacent epilogues where we remove extra kernel launches/memory traffic. For pure GEMM, use cuBLAS/CUTLASS/vLLM's existing kernel selection first.

This was tested against OPT-125M proxy shapes because no DeepSeek model/trace is currently present under the scratch model/cache/profile directories. A true DeepSeek-V4 run should repeat the same method using actual MLA/MLP shapes from the trace.

## What To Try With Helion

Good candidates:

- Fused activation+mul variants such as `silu_and_mul` and `gelu_tanh_and_mul`.
- RMSNorm variants, especially if Quack/Liger do not match the exact vLLM shape or dtype.
- Small elementwise fusions around residual add, RMSNorm, scaling, and gate/up projection epilogues.
- Shape-specific kernels discovered from DeepSeek MLA traces where no existing kernel library fits.

Poor first candidates:

- FlashAttention replacement. Use vLLM/FlashAttention/FlashInfer/TokenSpeed MLA first.
- Large GEMM replacement. Use cuBLAS/CUTLASS first unless the trace shows a very special small decode GEMM.
- Anything below roughly 2-3% of GPU time in Nsight.

## Development Pattern

1. Profile serving and identify a stable hot op family.
2. Build a tiny Helion harness matching the real tensor shape, dtype, layout, and numerical behavior.
3. Run with a quick autotune budget first.
4. Save the best config printed by Helion if the result is stable.
5. Compare against Torch, existing vLLM kernel, and any library candidate such as Liger/HF/Quack.
6. Only then wire it behind an env var in vLLM and re-run the serving matrix.

For vLLM integration, prefer an env-gated path first, for example `DEEPSEEK_PROF_USE_HELION_SILU_MUL=1`, so serving before/after comparisons stay clean.

## Scratch Hygiene

Use the normal scratch environment before every run:

```bash
source /scratch/deepseek-prof/env/scratch_env.sh
```

This keeps Triton, TorchInductor, CUDA, XDG, pip, and HF caches away from `~`. Helion also writes its generated code through the TorchInductor/Triton cache stack, so the scratch env matters.
