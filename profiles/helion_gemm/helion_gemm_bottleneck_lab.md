# Helion GEMM Bottleneck Lab

| shape `(M,K,N)` | torch mm | Helion mm | speedup | torch TFLOP/s | Helion TFLOP/s | first call s | max abs err | max rel err |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (8, 768, 3072) | 0.0112 | 0.0389 | 0.288 | 3.37 | 0.97 | 0.76 | 0 | 0 |
| (16, 768, 3072) | 0.0133 | 0.0386 | 0.345 | 5.67 | 1.96 | 0.23 | 0 | 0 |
| (32, 768, 3072) | 0.0121 | 0.0386 | 0.313 | 12.50 | 3.91 | 0.24 | 0 | 0 |
| (8, 3072, 768) | 0.0142 | 0.0388 | 0.367 | 2.65 | 0.97 | 0.24 | 0.125 | 1.48 |
| (16, 3072, 768) | 0.0163 | 0.0392 | 0.416 | 4.63 | 1.93 | 0.43 | 0.125 | 1.48 |
| (32, 3072, 768) | 0.0151 | 0.0388 | 0.389 | 10.02 | 3.89 | 0.29 | 0 | 0 |
| (8, 768, 768) | 0.0159 | 0.0383 | 0.413 | 0.60 | 0.25 | 0.24 | 0.0625 | 1.03 |
| (16, 768, 768) | 0.0163 | 0.0389 | 0.418 | 1.16 | 0.49 | 0.21 | 0.0625 | 1.97 |
| (32, 768, 768) | 0.0171 | 0.0396 | 0.431 | 2.21 | 0.95 | 0.24 | 0 | 0 |

## Shape Notes

- These are decode-linear proxy shapes, not a vLLM integration yet.
- `M` is the active token/request batch dimension; small `M` is the difficult decode case.
- `K=768,N=3072` and `K=3072,N=768` mimic OPT-125M MLP projections from the current local realistic workload.
- `K=768,N=768` mimics attention/output projections.
- A serving win requires replacing the actual vLLM path and re-running the realistic matrix.
