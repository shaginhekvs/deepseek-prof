| file | completed | request_throughput | output_throughput | total_token_throughput | mean_ttft_ms | median_ttft_ms | p90_ttft_ms | p95_ttft_ms | p99_ttft_ms | mean_tpot_ms | median_tpot_ms | p90_tpot_ms | p95_tpot_ms | p99_tpot_ms | mean_itl_ms | median_itl_ms | p90_itl_ms | p95_itl_ms | p99_itl_ms | mean_e2el_ms | median_e2el_ms | p90_e2el_ms | p95_e2el_ms | p99_e2el_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_heavy_c1_i128_o512_n24.json | 24 | 1.65 | 811.46 | 1024.93 | 7.01 | 7.04 | 7.64 | 7.84 | 8.05 | 1.22 | 1.22 | 1.22 | 1.22 | 1.22 | 1.22 | 1.22 | 1.24 | 1.25 | 1.28 | 604.95 | 607.69 | 714.09 | 722.05 | 722.67 |
| decode_heavy_c8_i128_o512_n24.json | 24 | 10.24 | 5026.80 | 6349.20 | 11.20 | 8.75 | 16.31 | 16.67 | 16.87 | 1.43 | 1.43 | 1.45 | 1.45 | 1.45 | 1.43 | 1.41 | 1.46 | 1.49 | 2.03 | 710.33 | 719.02 | 843.05 | 854.10 | 856.12 |
| mixed_chat_c1_i768_o256_n24.json | 24 | 3.37 | 801.49 | 3428.09 | 8.66 | 8.60 | 9.78 | 9.90 | 10.13 | 1.22 | 1.22 | 1.22 | 1.22 | 1.22 | 1.22 | 1.22 | 1.24 | 1.25 | 1.27 | 296.60 | 299.81 | 388.25 | 397.59 | 398.50 |
| mixed_chat_c8_i768_o256_n24.json | 24 | 17.28 | 4109.58 | 17577.30 | 14.50 | 11.84 | 23.15 | 23.74 | 23.84 | 1.63 | 1.62 | 1.75 | 1.75 | 1.77 | 1.63 | 1.63 | 1.79 | 1.92 | 2.91 | 399.57 | 395.37 | 530.47 | 539.13 | 557.48 |
| prefill_heavy_c1_i1536_o32_n24.json | 24 | 19.72 | 610.56 | 31094.31 | 15.20 | 15.51 | 16.83 | 16.89 | 17.02 | 1.18 | 1.18 | 1.19 | 1.20 | 1.21 | 1.18 | 1.22 | 1.25 | 1.26 | 1.37 | 50.62 | 50.78 | 55.90 | 57.98 | 59.11 |
| prefill_heavy_c8_i1536_o32_n24.json | 24 | 90.16 | 2791.31 | 142153.89 | 21.66 | 17.85 | 37.95 | 39.26 | 39.68 | 2.03 | 2.09 | 2.15 | 2.17 | 2.21 | 2.02 | 2.02 | 2.88 | 3.42 | 3.99 | 82.27 | 80.29 | 100.80 | 104.20 | 106.69 |

## Bottleneck Reading
- High `mean_ttft_ms` with low `mean_tpot_ms`: prefill/model-load/scheduler admission pressure.
- Low TTFT but high `mean_tpot_ms`/`mean_itl_ms`: decode, sampling, KV-cache, or GPU launch overhead.
- Latency grows sharply from concurrency 1 to 8/32: saturation or queueing.
- Output throughput plateaus while latency rises: GPU is saturated; inspect Nsight kernels next.
- Request throughput low but GPU kernels sparse: frontend/tokenization/HTTP/scheduler CPU bottleneck.
