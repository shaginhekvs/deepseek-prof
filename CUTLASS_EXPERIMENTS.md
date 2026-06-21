# CUTLASS Experiments

CUTLASS is highly relevant to this project because the Nsight decode-heavy trace already showed CUTLASS GEMM kernels in the vLLM hot path. The practical question is not whether CUTLASS matters, but which GEMM shapes and epilogues are worth specializing.

Sources:

- `https://github.com/NVIDIA/cutlass`
- `https://docs.nvidia.com/cutlass/index.html`
- `https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html`

## Local Setup

The Python package already present in the vLLM environment is:

- `nvidia-cutlass-dsl==4.5.2`

The upstream CUTLASS repo is cloned for examples:

- `/scratch/deepseek-prof/src/cutlass`

The current lab uses the Ampere CuTe DSL `TensorOpGemm` example path, which runs on the local A10 (`sm_86`).

## Run

```bash
source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate
python /scratch/deepseek-prof/scripts/cutlass_gemm_lab.py --warmup 10 --iters 50
```

Artifacts:

- `/scratch/deepseek-prof/profiles/cutlass/cutlass_gemm_lab.json`
- `/scratch/deepseek-prof/profiles/cutlass/cutlass_gemm_lab.md`

## Current Result

Plain fp16 BMM, compared against `torch.bmm`:

- `(B=2, M=512, N=512, K=256)`: CUTLASS `1.21x` faster.
- `(B=8, M=128, N=128, K=512)`: CUTLASS `0.76x` vs `torch.bmm`.
- `(B=32, M=64, N=64, K=512)`: CUTLASS `0.78x` vs `torch.bmm`.

Interpretation: the simple CUTLASS example is competitive, but not automatically better than cuBLAS-backed PyTorch on small batched GEMMs. For vLLM decode, the interesting CUTLASS work is likely:

- fused epilogues, such as bias/activation/residual or scaling fused into GEMM output;
- shape-specific kernels for the exact small decode GEMM shapes found in Nsight;
- grouped GEMM if the model path has many same-ish small expert/projection GEMMs;
- comparing against vLLM's existing CUTLASS kernel selection, not just against raw PyTorch.

## How This Compares To Helion

Use Helion for fast iteration on fused elementwise kernels and epilogues. Use CUTLASS when the hot operation is a GEMM or GEMM-like contraction and you need Tensor Core control.

Current local picture:

- Helion fused `silu_and_mul` beats raw Torch by about `1.74-1.85x`.
- CUTLASS plain fp16 BMM beats `torch.bmm` on one larger shape but loses on smaller batched shapes.
- Quack RMSNorm, which is also CUTLASS/CuTe-adjacent, is currently the best RMSNorm microbenchmark candidate.

## Next CUTLASS Work

1. Extract exact GEMM shapes from the vLLM/DeepSeek trace.
2. Re-run `cutlass_gemm_lab.py --shapes ...` with those shapes.
3. If the plain GEMM loses, do not stop there: test fused epilogues or grouped GEMM.
4. Use Nsight Compute on the candidate CUTLASS kernel to inspect occupancy, Tensor Core usage, shared memory, register pressure, and memory throughput.
5. Only integrate into vLLM behind an env flag after the standalone kernel beats the existing path for the exact shape.
