# DeepSeek Fusion Experiments

This is the next stage after the `fused_add_rms_norm` Helion provider: test more fusion opportunities that are actually in the vLLM DeepSeek pipeline.

Run:

```bash
source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate
cd /scratch/deepseek-prof
python scripts/bench_deepseek_pipeline_fusions.py --warmup 10 --iters 50
```

Targets:

- Dense MLP `SiluAndMul`: `DeepseekV2MLP.forward` calls `self.act_fn(gate_up)`.
- MLA FP16 overflow path: `DeepseekV2DecoderLayer.forward` scales `hidden_states` and `residual` before `post_attention_layernorm`.
- MoE backend probe: checks whether FlashInfer CuTeDSL/CUTLASS-style MoE paths are usable on this A10 host.

Outputs:

- `/scratch/deepseek-prof/profiles/deepseek_fusions/deepseek_pipeline_fusions.md`
- `/scratch/deepseek-prof/profiles/deepseek_fusions/deepseek_pipeline_fusions.json`

Current A10 findings:

| area | best result | decision |
|---|---|---|
| Dense MLP `SiluAndMul` | HF is best at `128x2048`; Helion is only ~1.01x faster than vLLM C on large shapes. | Do not replace globally. Consider an env-gated HF small-batch experiment only. |
| MLA `scale + residual RMSNorm` | Helion fused is `3.10x` faster at `1024x3072` and `3.09x` faster at `8192x3072` versus separate scale plus vLLM C. | Best integration candidate. |
| CuTeDSL MoE | CUTLASS DSL is installed, but vLLM FlashInfer CuTeDSL MoE is an SM100/NVFP4-oriented path; this host is A10 SM86. | Not an A10 DeepSeek MoE target unless building a separate custom CUTLASS experiment. |

Latest table:

```text
SiluAndMul:
128x2048   torch 0.0243 ms  vLLM C 0.0123 ms  HF 0.0053 ms  Helion 0.0356 ms
1024x2048  torch 0.0504 ms  vLLM C 0.0276 ms  HF 0.0290 ms  Helion 0.0342 ms
1024x4096  torch 0.0964 ms  vLLM C 0.0540 ms  HF 0.0548 ms  Helion 0.0535 ms
8192x3072  torch 0.5375 ms  vLLM C 0.3109 ms  HF 0.3204 ms  Helion 0.3075 ms

Scale + residual RMSNorm:
128x3072   torch 0.1077 ms  scale+vLLM C 0.0389 ms  Helion fused 0.0419 ms
1024x3072  torch 0.5219 ms  scale+vLLM C 0.1640 ms  Helion fused 0.0529 ms
8192x3072  torch 3.9804 ms  scale+vLLM C 1.2566 ms  Helion fused 0.4070 ms
```
