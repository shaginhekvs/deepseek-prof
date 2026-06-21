# Helion Fusion Bottleneck Lab

| op | shape | baseline ms | Helion ms | speedup | first call s | max abs err | max rel err | HF+add ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| silu_mul_residual | (1024, 768) | 0.0270 | 0.0366 | 0.737 | 0.72 | 0.00781 | 1.72 | 0.0190 |
| add_rms_norm | (1024, 768) | 0.0531 | 0.0382 | 1.392 | 0.46 | 0 | 0 |  |
| silu_mul_add_rms_norm | (1024, 768) | 0.0741 |  |  |  |  |  |  |
| silu_mul_residual | (8192, 768) | 0.2193 | 0.0997 | 2.199 | 0.09 | 0.00781 | 1.97 | 0.1555 |
| add_rms_norm | (8192, 768) | 0.2810 | 0.0784 | 3.584 | 0.13 | 0 | 0 |  |
| silu_mul_add_rms_norm | (8192, 768) | 0.4216 |  |  |  |  |  |  |
| silu_mul_residual | (1024, 3072) | 0.1164 | 0.0538 | 2.162 | 0.16 | 0.0156 | 1.83 | 0.0816 |
| add_rms_norm | (1024, 3072) | 0.1430 | 0.0408 | 3.503 | 0.43 | 0 | 0 |  |
| silu_mul_add_rms_norm | (1024, 3072) | 0.2190 |  |  |  |  |  |  |
| silu_mul_residual | (8192, 3072) | 0.8488 | 0.4021 | 2.111 | 0.10 | 0.0156 | 1.95 | 0.6258 |
| add_rms_norm | (8192, 3072) | 1.0717 | 0.2967 | 3.612 | 0.13 | 0 | 0 |  |
| silu_mul_add_rms_norm | (8192, 3072) | 1.6121 |  |  |  |  |  |  |

## Helion Failures

- `silu_mul_add_rms_norm` shape `(1024, 768)`: `NoConfigFound('No working config found from autotuning')`
- `silu_mul_add_rms_norm` shape `(8192, 768)`: `NoConfigFound('No working config found from autotuning')`
- `silu_mul_add_rms_norm` shape `(1024, 3072)`: `NoConfigFound('No working config found from autotuning')`
- `silu_mul_add_rms_norm` shape `(8192, 3072)`: `NoConfigFound('No working config found from autotuning')`

## Notes

- These are fusion candidates around the current GEMM/MLP/norm bottleneck area, not replacements for GEMM itself.
- `silu_mul_residual` tests whether Helion can beat separate activation/gate plus residual add.
- `add_rms_norm` tests a common residual-plus-normalization fusion.
- `silu_mul_add_rms_norm` tests a larger fused chain that removes multiple intermediate writes.
- A vLLM integration should only follow for a row where Helion beats both raw Torch and the existing library path.
