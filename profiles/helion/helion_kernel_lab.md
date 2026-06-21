# Helion Kernel Lab

| op | shape | torch ms | helion ms | helion speedup | helion first call s | hf ms | hf speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| silu_and_mul | (1024, 3072) | 0.0759 | 0.0410 | 1.851 | 0.60 | 0.0420 | 1.806 |
| silu_and_mul | (8192, 3072) | 0.5382 | 0.3092 | 1.740 | 53.92 | 0.3131 | 1.719 |

## Notes

- Helion first-call time includes compilation and quick autotuning.
- Compare Helion against both Torch and HF activation because HF is already a strong fused kernel candidate.
- Autotune logs are written under `/scratch/deepseek-prof/profiles/helion/`.
