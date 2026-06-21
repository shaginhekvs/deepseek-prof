# DeepSeek Helion Integration

The first real DeepSeek pipeline target is vLLM's shared `fused_add_rms_norm` IR op, not a standalone toy kernel.

Why this op:

- `vllm/model_executor/models/deepseek_v2.py` uses `RMSNorm` in the decoder input norm, post-attention norm, final norm, and MLA LoRA-rank norms.
- The residual decoder norms call `RMSNorm.forward_native(..., residual=...)`.
- That routes to `ir.ops.fused_add_rms_norm.maybe_inplace(...)`.
- Therefore a provider registered for `fused_add_rms_norm` is exercised by DeepSeek decoder layers through the normal vLLM path.

Local integration:

- Provider file: `/scratch/deepseek-prof/src/vllm/vllm/kernels/helion_ops.py`
- Env switch: `VLLM_USE_HELION_OPS=1`
- Platform priority hook: CUDA prepends `helion` for `fused_add_rms_norm` only.
- Benchmark harness: `/scratch/deepseek-prof/scripts/bench_vllm_deepseek_helion_rmsnorm.py`
- Results path: `/scratch/deepseek-prof/profiles/helion_vllm_ir/`

Current A10 result:

| shape | native ms | vLLM C ms | Helion ms | Helion vs native | Helion vs vLLM C |
|---|---:|---:|---:|---:|---:|
| 128x576 | 0.1040 | 0.0075 | 0.0607 | 1.72x | 0.12x |
| 1024x576 | 0.1035 | 0.0075 | 0.0596 | 1.74x | 0.13x |
| 128x3072 | 0.1035 | 0.0075 | 0.0603 | 1.72x | 0.12x |
| 1024x3072 | 0.4145 | 0.0544 | 0.0580 | 7.15x | 0.94x |
| 8192x3072 | 3.1414 | 0.4181 | 0.4083 | 7.69x | 1.02x |

Interpretation: Helion is not useful for the small MLA-rank shapes yet because vLLM C is far faster there. It is competitive on larger hidden-size norms and slightly faster than vLLM C for the current `8192x3072` proxy. That makes it worth trying under real DeepSeek serving traces, especially when the compiled path would otherwise use the native composite.

Run:

```bash
source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate
cd /scratch/deepseek-prof
python scripts/bench_vllm_deepseek_helion_rmsnorm.py --warmup 10 --iters 50
```

Try inside vLLM serving:

```bash
export VLLM_USE_HELION_OPS=1
export VLLM_LOGGING_LEVEL=DEBUG
# then launch the DeepSeek/vLLM workload normally
```

Current safety limits:

- Functional provider, not in-place. This preserves correctness while testing.
- CUDA fp16 only.
- No `variance_size` override.
- Contiguous/viewable 2D-like tensors only.
- At least 128 rows. Smaller decode batches fall back to the next provider.
- Hidden sizes restricted to DeepSeek-like hidden/rank shapes.

Next kernel-development steps:

1. Compare this provider against vLLM C on exact DeepSeek serving traces.
2. If Helion is competitive, build an in-place provider to match `torch.ops._C.fused_add_rms_norm` memory behavior.
3. Extend the same process to MLA-specific kernels after extracting exact vLLM DeepSeek MLA tensors.
