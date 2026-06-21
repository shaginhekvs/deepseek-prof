# GEMM Backend Compare

| shape `(M,K,N)` | torch mm | Helion | CuTeDSL | Helion speedup | CuTeDSL speedup | torch TFLOP/s | Helion TFLOP/s | CuTeDSL TFLOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (8, 768, 3072) | 0.0113 | 0.0405 | 0.0182 | 0.278 | 0.619 | 3.35 | 0.93 | 2.08 |
| (16, 768, 3072) | 0.0112 | 0.0398 | 0.0182 | 0.283 | 0.617 | 6.71 | 1.90 | 4.14 |
| (32, 768, 3072) | 0.0108 | 0.0397 | 0.0184 | 0.272 | 0.588 | 13.99 | 3.80 | 8.23 |
| (8, 3072, 768) | 0.0141 | 0.0398 | 0.0628 | 0.356 | 0.225 | 2.67 | 0.95 | 0.60 |
| (16, 3072, 768) | 0.0146 | 0.0401 | 0.0628 | 0.363 | 0.232 | 5.19 | 1.88 | 1.20 |
| (32, 3072, 768) | 0.0148 | 0.0393 | 0.0629 | 0.377 | 0.235 | 10.20 | 3.84 | 2.40 |
| (8, 768, 768) | 0.0145 | 0.0400 | 0.0180 | 0.364 | 0.806 | 0.65 | 0.24 | 0.52 |
| (16, 768, 768) | 0.0149 | 0.0397 | 0.0180 | 0.375 | 0.825 | 1.27 | 0.48 | 1.05 |
| (32, 768, 768) | 0.0156 | 0.0398 | 0.0182 | 0.392 | 0.858 | 2.42 | 0.95 | 2.08 |

## Error Check

| shape `(M,K,N)` | Helion abs | Helion rel | CuTeDSL abs | CuTeDSL rel |
| ---: | ---: | ---: | ---: | ---: |
| (8, 768, 3072) | 0 | 0 | 0 | 0 |
| (16, 768, 3072) | 0 | 0 | 0 | 0 |
| (32, 768, 3072) | 0 | 0 | 0 | 0 |
| (8, 3072, 768) | 0.125 | 1.48 | 0.125 | 1.48 |
| (16, 3072, 768) | 0.125 | 1.48 | 0.125 | 1.48 |
| (32, 3072, 768) | 0 | 0 | 0 | 0 |
| (8, 768, 768) | 0.0625 | 1.03 | 0.0625 | 1.03 |
| (16, 768, 768) | 0.0625 | 1.97 | 0.0625 | 1.97 |
| (32, 768, 768) | 0 | 0 | 0 | 0 |

## Notes

- CuTeDSL is run through the CUTLASS Ampere `TensorOpGemm` BMM example with batch size `1`.
- These are standalone proxy shapes for the current vLLM decode-linear bottleneck, not a vLLM integration.
- For real serving impact, replace a traced vLLM op behind an env flag and re-run the realistic matrix.
