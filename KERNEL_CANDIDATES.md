# Kernel Candidate Experiments

The current decode-heavy Nsight trace points at these targets:

1. FlashAttention split-KV.
2. Small decode GEMMs.
3. Metadata/fill/slot-mapping overhead.
4. Layernorm/RMSNorm/SwiGLU/RoPE style fused elementwise kernels for DeepSeek-like models.

## Candidate Libraries

### ThunderKittens

- Source: `https://github.com/HazyResearch/ThunderKittens`
- Local clone: `/scratch/deepseek-prof/src/ThunderKittens`
- Role: custom CUDA kernel authoring for attention/GEMM-like kernels.
- Not a drop-in Python import for vLLM.
- Best use here: once we know the exact DeepSeek-V4 MLA or small-GEMM shape, write a focused CUDA/TK prototype and compare with Nsight Compute.
- Priority: medium/high for a custom kernel project, low for an immediate serving win. Use it after extracting exact tensor shapes from a DeepSeek/vLLM trace.

### Liger Kernel

- Source: `https://github.com/linkedin/Liger-Kernel`
- Installed package: `liger-kernel`
- Role: Triton kernels for RMSNorm, RoPE, SwiGLU, softmax, fused add RMSNorm, losses, and related transformer ops.
- Mostly aimed at training, but useful for isolated inference-adjacent op microbenchmarks.
- Best use here: compare RMSNorm/RoPE/SwiGLU/softmax kernels against Torch/vLLM/Inductor variants. For DeepSeek, RMSNorm/RoPE/SwiGLU are more relevant than OPT's LayerNorm.
- Local result: useful for larger RMSNorm and SwiGLU shapes, not useful for OPT-like LayerNorm or vocab softmax on this A10 run.

### Hugging Face Kernel Hub

- Source: `https://github.com/huggingface/kernels`
- Installed package: `kernels`
- Role: load pinned hub kernels with `get_kernel(repo, version=...)`.
- Best use here: test hub kernels as importable modules first, then microbenchmark specific callables if the repo exposes a matching op.
- Local result: `kernels-community/activation` loaded and fused activation+mul kernels were good microbench wins. `kernels-community/triton_kernels` returned 401 without auth.

### Quack Kernels

- Installed package: `quack-kernels`
- Role: CUTLASS/Cute-DSL style kernels exposed as Python ops.
- Best use here: RMSNorm is the current best microbenchmark candidate. Its softmax failed at the tested vocab shape on A10 because the generated kernel exceeded the `sm_86` shared-memory launch limit.

### TokenSpeed MLA

- Installed package: `tokenspeed_mla`
- Role: MLA prefill/decode kernels and FP8 KV packing.
- Best use here: high-priority DeepSeek experiment. The public API exposes `tokenspeed_mla_prefill`, `tokenspeed_mla_decode`, and `mla_kv_pack_quantize_fp8`, so the next serious experiment is to match those inputs to vLLM's DeepSeek MLA tensors and compare against the current vLLM attention path.

### Already Installed vLLM-Adjacent Kernels

- `flashinfer-python`
- `humming-kernels`
- `quack-kernels`
- `tokenspeed-triton`
- `tokenspeed_mla`
- `nvidia-cutlass-dsl`

These are especially relevant because vLLM may already integrate or compete with them. `tokenspeed_mla` is interesting for MLA-like DeepSeek paths.

## Run

```bash
source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate
python /scratch/deepseek-prof/scripts/kernel_candidate_microbench.py
```

Artifacts:

- `/scratch/deepseek-prof/profiles/kernel_candidates/kernel_candidate_microbench.json`
- `/scratch/deepseek-prof/profiles/kernel_candidates/kernel_candidate_microbench.md`

## Current Microbench Takeaways

Short run on the local A10 with fp16 tensors:

- `quack.rmsnorm`: best RMSNorm candidate for larger shapes, up to about `3.6x` faster than the Torch baseline at `(8192, 3072)`.
- `liger_rms_norm`: useful for larger shapes, about `1.8x` faster at `(8192, 768)` and `(8192, 3072)`, but slower at `(1024, 768)`.
- `liger_swiglu`: consistently useful in this sweep, about `1.6x` faster at `(1024, 3072)` and `(8192, 3072)`.
- `kernels-community/activation`: fused `silu_and_mul` and `gelu_tanh_and_mul` were about `1.7x` faster than separate Torch activation+mul.
- `liger_softmax`: slower than Torch for the tested vocab-ish shapes.
- `quack.softmax`: failed to launch for `(128/1024, 50272)` on `sm_86` due shared-memory requirements, so it is not a candidate for this server without changing shape/kernel config.

This is still a microbench result. The first serving-level candidates should be RMSNorm/SwiGLU or MLA, not softmax.

## Integration Order

1. Re-run the serving workload with a DeepSeek-like model path and capture exact RMSNorm/SwiGLU/MLA shapes from Nsight or PyTorch profiler.
2. Prototype a narrow vLLM patch that swaps only RMSNorm first, guarded behind an env var.
3. Repeat the same realistic serving matrix and compare TTFT, TPOT, output tokens/sec, and Nsight kernel share.
4. Try fused activation/SwiGLU next if the model path exposes an unfused Torch/Triton implementation.
5. Build the TokenSpeed MLA input harness from vLLM DeepSeek tensors. This is the most domain-relevant path for DeepSeek support work.
6. Use ThunderKittens only after the traces show a stable hot shape that the drop-in libraries do not cover.

## Interpretation Rules

- A microbenchmark win does not automatically mean vLLM serving improves.
- If an op is below 2-3% of Nsight GPU time, it is not the first integration target.
- Prefer kernels that match a known hot family from Nsight: attention, small GEMM, KV update, slot/block metadata, sampling.
- For anything that replaces a vLLM kernel, run the serving matrix before/after with the same `TEMPERATURE`, prompts, concurrency, and cache state.
