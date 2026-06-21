# CUTLASS GEMM Lab

| shape `(B,M,N,K)` | torch.bmm ms | CUTLASS ms | speedup | torch TFLOP/s | CUTLASS TFLOP/s | compile s | first call s | max rel err |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (2, 512, 512, 256) | 0.0110 | 0.0092 | 1.206 | 24.32 | 29.32 | 1.34 | 0.0005 | 0 |
| (8, 128, 128, 512) | 0.0105 | 0.0138 | 0.763 | 12.75 | 9.72 | 1.16 | 0.0002 | 0 |
| (32, 64, 64, 512) | 0.0103 | 0.0133 | 0.781 | 12.98 | 10.13 | 1.15 | 0.0003 | 0 |

## Notes

- This uses NVIDIA CUTLASS CuTe DSL `TensorOpGemm` from the cloned upstream examples.
- `torch.bmm` is usually cuBLAS-backed, so this comparison is CUTLASS vs a strong vendor library baseline.
- The current kernel is a plain fp16 BMM. For vLLM wins, the more interesting CUTLASS work is fused epilogues or shape-specialized small decode GEMMs.
