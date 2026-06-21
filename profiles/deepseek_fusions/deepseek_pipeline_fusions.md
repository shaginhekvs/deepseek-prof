# DeepSeek Pipeline Fusion Experiments

## SiluAndMul

| shape | torch ms | vLLM C ms | HF ms | Helion ms | Helion vs vLLM C | best |
|---|---:|---:|---:|---:|---:|---|
| 128x2048 | 0.0240 | 0.0126 | 0.0052 | 0.0348 | 0.36x | HF |
| 1024x2048 | 0.0503 | 0.0276 | 0.0290 | 0.0342 | 0.81x | vLLM C |
| 1024x4096 | 0.0964 | 0.0540 | 0.0548 | 0.0536 | 1.01x | Helion |
| 8192x3072 | 0.5390 | 0.3111 | 0.3195 | 0.3076 | 1.01x | Helion |

## Scale Plus Residual RMSNorm

| shape | torch ms | scale+vLLM C ms | Helion fused ms | Helion vs scale+vLLM C | best |
|---|---:|---:|---:|---:|---|
| 128x3072 | 0.1081 | 0.0397 | 0.0412 | 0.96x | scale+vLLM C |
| 1024x3072 | 0.5238 | 0.1637 | 0.0531 | 3.09x | Helion fused |
| 8192x3072 | 3.9814 | 1.2563 | 0.4057 | 3.10x | Helion fused |

## MoE CuTeDSL/CUTLASS Probe

- `{"capability": [8, 6], "gpu": "NVIDIA A10", "section": "deepseek_moe_backend_probe"}`
- `{"backend": "flashinfer_cutedsl_moe_nvfp4", "current_device_supported": false, "note": "vLLM requires CUDA SM100 family for this provider.", "section": "deepseek_moe_backend_probe", "symbol_available": true}`
- `{"backend": "flashinfer_cutedsl_grouped_gemm_nt_masked", "current_device_supported": false, "note": "Used by FlashInfer CuteDSL batched MoE paths when supported.", "section": "deepseek_moe_backend_probe", "symbol_available": true}`
- `{"available": true, "backend": "cutlass_dsl_python", "module": "/scratch/deepseek-prof/env/py312/lib64/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/__init__.py", "note": "Available for custom GEMM labs; vLLM MoE CUTLASS paths are quantization/backend-specific.", "section": "deepseek_moe_backend_probe"}`

Interpretation:
- `SiluAndMul` is already a vLLM C custom op in DeepSeek dense MLP layers; replace it only if a candidate beats that kernel, not just raw Torch.
- `scale + residual RMSNorm` is a real DeepSeek MLA FP16 branch and is a plausible Helion fusion target because vLLM currently performs scaling separately before the fused RMSNorm.
- FlashInfer CuTeDSL MoE is mainly an NVFP4/SM100-family path in this vLLM tree; on A10 it is expected to be unavailable.
