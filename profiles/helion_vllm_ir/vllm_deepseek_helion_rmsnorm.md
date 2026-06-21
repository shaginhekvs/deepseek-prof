# vLLM DeepSeek Helion RMSNorm Experiment

This benchmarks vLLM's actual `fused_add_rms_norm` IR provider used by DeepSeek decoder residual RMSNorm.

| shape | native ms | vllm_c ms | helion ms | helion vs native | helion vs vllm_c | first call s | out abs err | residual abs err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128x576 | 0.1040 | 0.0075 | 0.0607 | 1.72x | 0.12x | 0.68 | 0.007812 | 0 |
| 1024x576 | 0.1035 | 0.0075 | 0.0596 | 1.74x | 0.13x | 0.34 | 0.007812 | 0 |
| 128x3072 | 0.1035 | 0.0075 | 0.0603 | 1.72x | 0.12x | 0.25 | 0.007812 | 0 |
| 1024x3072 | 0.4145 | 0.0544 | 0.0580 | 7.15x | 0.94x | 0.35 | 0.01562 | 0 |
| 8192x3072 | 3.1414 | 0.4181 | 0.4083 | 7.69x | 1.02x | 0.31 | 0.01562 | 0 |
