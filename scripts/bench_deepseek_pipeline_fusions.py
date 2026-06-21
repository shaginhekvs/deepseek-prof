#!/usr/bin/env python3
"""DeepSeek-path fusion experiments for vLLM on A10.

This script focuses on operations that are actually present in
vllm/model_executor/models/deepseek_v2.py:

* DeepseekV2MLP: gate_up_proj -> SiluAndMul -> down_proj.
* DeepseekV2DecoderLayer FP16 MLA path: scale hidden/residual, then
  post_attention_layernorm(hidden_states, residual).
* DeepseekV2MoE: report whether CuTeDSL/CUTLASS-style MoE paths are available
  on the current GPU.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import helion
import helion.language as hl
import vllm.kernels  # noqa: F401
from vllm.kernels import vllm_c


ROOT = Path("/scratch/deepseek-prof")
OUT = ROOT / "profiles" / "deepseek_fusions"
OUT.mkdir(parents=True, exist_ok=True)


def cuda_time_ms(fn: Callable, warmup: int, iters: int):
    y = None
    for _ in range(warmup):
        y = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        y = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, y


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().detach().cpu())


def max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float()
    bf = b.float()
    denom = torch.maximum(af.abs(), bf.abs()).clamp_min(1e-5)
    return float(((af - bf).abs() / denom).max().detach().cpu())


def torch_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[:, :d]) * x[:, d:]


def vllm_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((x.shape[0], x.shape[1] // 2), device=x.device, dtype=x.dtype)
    torch.ops._C.silu_and_mul(out, x)
    return out


def load_hf_activation():
    try:
        from kernels import get_kernel

        return get_kernel("kernels-community/activation", version=1, trust_remote_code=True)
    except Exception:
        return None


@helion.kernel(
    static_shapes=True,
    autotune_effort="quick",
    autotune_budget_seconds=45,
    autotune_log=str(OUT / "silu_and_mul_autotune"),
    autotune_baseline_fn=torch_silu_and_mul,
    autotune_baseline_atol=1e-2,
    autotune_baseline_rtol=1e-2,
)
def helion_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    rows = x.size(0)
    hidden = x.size(1) // 2
    out = torch.empty((rows, hidden), device=x.device, dtype=x.dtype)

    for tile_m, tile_n in hl.tile([rows, hidden]):
        gate = x[tile_m, tile_n]
        up = x[tile_m, tile_n + hidden]
        out[tile_m, tile_n] = gate * torch.sigmoid(gate) * up

    return out


def torch_scaled_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = x.float().mul(scale) + residual.float().mul(scale)
    residual_out = y.to(x.dtype)
    out = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = out.to(weight.dtype) * weight
    return out.to(x.dtype), residual_out


def vllm_scaled_then_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    xi = x.clone()
    ri = residual.clone()
    xi.mul_(scale)
    ri.mul_(scale)
    return vllm_c.fused_add_rms_norm.impl_fn(xi, ri, weight, eps)


@helion.kernel(
    static_shapes=True,
    autotune_effort="quick",
    autotune_budget_seconds=45,
    autotune_log=str(OUT / "scaled_add_rms_norm_autotune"),
    autotune_baseline_fn=torch_scaled_add_rms_norm,
    autotune_baseline_atol=2e-2,
    autotune_baseline_rtol=2e-2,
)
def helion_scaled_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = x.size(0)
    out = torch.empty_like(x)
    residual_out = torch.empty_like(residual)

    for tile_m in hl.tile(rows):
        y = (x[tile_m, :] + residual[tile_m, :]) * scale
        residual_out[tile_m, :] = y
        variance = (y * y).mean(dim=-1, keepdim=True)
        out[tile_m, :] = y * torch.rsqrt(variance + eps) * weight[None, :]

    return out, residual_out


def bench_silu_and_mul(shape: str, warmup: int, iters: int, hf_activation) -> dict:
    rows, hidden = [int(x) for x in shape.lower().split("x")]
    dtype = torch.float16
    torch.manual_seed(rows * 100000 + hidden)
    x = torch.randn((rows, hidden * 2), device="cuda", dtype=dtype)

    torch_ms, torch_y = cuda_time_ms(lambda: torch_silu_and_mul(x), warmup, iters)
    vllm_ms, vllm_y = cuda_time_ms(lambda: vllm_silu_and_mul(x), warmup, iters)

    row = {
        "section": "deepseek_mlp_silu_and_mul",
        "shape": [rows, hidden],
        "torch_ms": torch_ms,
        "vllm_c_ms": vllm_ms,
        "vllm_c_speedup_vs_torch": torch_ms / vllm_ms if vllm_ms else None,
        "vllm_c_abs_error": max_abs(torch_y, vllm_y),
    }

    if hf_activation is not None:
        out = torch.empty_like(vllm_y)

        def hf_impl():
            hf_activation.silu_and_mul(out, x)
            return out

        hf_ms, hf_y = cuda_time_ms(hf_impl, warmup, iters)
        row.update(
            {
                "hf_ms": hf_ms,
                "hf_speedup_vs_torch": torch_ms / hf_ms if hf_ms else None,
                "hf_speedup_vs_vllm_c": vllm_ms / hf_ms if hf_ms else None,
                "hf_abs_error": max_abs(torch_y, hf_y),
            }
        )

    try:
        first = time.time()
        helion_first_y = helion_silu_and_mul(x)
        torch.cuda.synchronize()
        first_call_s = time.time() - first
        helion_ms, helion_y = cuda_time_ms(lambda: helion_silu_and_mul(x), warmup, iters)
        row.update(
            {
                "helion_ms": helion_ms,
                "helion_first_call_s": first_call_s,
                "helion_speedup_vs_torch": torch_ms / helion_ms if helion_ms else None,
                "helion_speedup_vs_vllm_c": vllm_ms / helion_ms if helion_ms else None,
                "helion_abs_error": max_abs(torch_y, helion_y),
                "helion_rel_error": max_rel(torch_y, helion_y),
                "helion_first_abs_error": max_abs(torch_y, helion_first_y),
            }
        )
    except Exception as exc:
        row["helion_error"] = repr(exc)
    return row


def bench_scaled_add_norm(shape: str, warmup: int, iters: int) -> dict:
    rows, hidden = [int(x) for x in shape.lower().split("x")]
    dtype = torch.float16
    eps = 1e-5
    scale = 1.0 / 16.0
    torch.manual_seed(rows * 100000 + hidden + 17)
    x = torch.randn((rows, hidden), device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn((hidden,), device="cuda", dtype=dtype)

    torch_ms, torch_y = cuda_time_ms(
        lambda: torch_scaled_add_rms_norm(x, residual, weight, eps, scale),
        warmup,
        iters,
    )
    vllm_ms, vllm_y = cuda_time_ms(
        lambda: vllm_scaled_then_fused_add_rms_norm(x, residual, weight, eps, scale),
        warmup,
        iters,
    )
    row = {
        "section": "deepseek_mla_scale_plus_post_attention_rmsnorm",
        "shape": [rows, hidden],
        "torch_ms": torch_ms,
        "vllm_scale_then_c_ms": vllm_ms,
        "vllm_scale_then_c_speedup_vs_torch": torch_ms / vllm_ms if vllm_ms else None,
        "vllm_out_abs_error": max_abs(torch_y[0], vllm_y[0]),
        "vllm_residual_abs_error": max_abs(torch_y[1], vllm_y[1]),
    }

    try:
        first = time.time()
        helion_first_y = helion_scaled_add_rms_norm(x, residual, weight, eps, scale)
        torch.cuda.synchronize()
        first_call_s = time.time() - first
        helion_ms, helion_y = cuda_time_ms(
            lambda: helion_scaled_add_rms_norm(x, residual, weight, eps, scale),
            warmup,
            iters,
        )
        row.update(
            {
                "helion_ms": helion_ms,
                "helion_first_call_s": first_call_s,
                "helion_speedup_vs_torch": torch_ms / helion_ms if helion_ms else None,
                "helion_speedup_vs_vllm_scale_then_c": vllm_ms / helion_ms
                if helion_ms
                else None,
                "helion_out_abs_error": max_abs(torch_y[0], helion_y[0]),
                "helion_residual_abs_error": max_abs(torch_y[1], helion_y[1]),
                "helion_first_out_abs_error": max_abs(torch_y[0], helion_first_y[0]),
            }
        )
    except Exception as exc:
        row["helion_error"] = repr(exc)
    return row


def probe_moe_backends() -> list[dict]:
    rows = []
    props = torch.cuda.get_device_properties(0)
    sm100_family = props.major >= 10
    rows.append(
        {
            "section": "deepseek_moe_backend_probe",
            "gpu": props.name,
            "capability": [props.major, props.minor],
        }
    )
    try:
        from vllm.utils.flashinfer import (
            has_flashinfer_cutedsl_grouped_gemm_nt_masked,
            has_flashinfer_cutedsl_moe_nvfp4,
        )

        rows.append(
            {
                "section": "deepseek_moe_backend_probe",
                "backend": "flashinfer_cutedsl_moe_nvfp4",
                "symbol_available": bool(has_flashinfer_cutedsl_moe_nvfp4()),
                "current_device_supported": sm100_family,
                "note": "vLLM requires CUDA SM100 family for this provider.",
            }
        )
        rows.append(
            {
                "section": "deepseek_moe_backend_probe",
                "backend": "flashinfer_cutedsl_grouped_gemm_nt_masked",
                "symbol_available": bool(has_flashinfer_cutedsl_grouped_gemm_nt_masked()),
                "current_device_supported": sm100_family,
                "note": "Used by FlashInfer CuteDSL batched MoE paths when supported.",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "section": "deepseek_moe_backend_probe",
                "backend": "flashinfer_cutedsl",
                "available": False,
                "error": repr(exc),
            }
        )
    try:
        import cutlass

        rows.append(
            {
                "section": "deepseek_moe_backend_probe",
                "backend": "cutlass_dsl_python",
                "available": True,
                "module": getattr(cutlass, "__file__", None),
                "note": "Available for custom GEMM labs; vLLM MoE CUTLASS paths are quantization/backend-specific.",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "section": "deepseek_moe_backend_probe",
                "backend": "cutlass_dsl_python",
                "available": False,
                "error": repr(exc),
            }
        )
    return rows


def write_markdown(results: list[dict], path: Path) -> None:
    lines = ["# DeepSeek Pipeline Fusion Experiments", ""]
    lines.append("## SiluAndMul")
    lines.append("")
    lines.append("| shape | torch ms | vLLM C ms | HF ms | Helion ms | Helion vs vLLM C | best |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for r in results:
        if r.get("section") != "deepseek_mlp_silu_and_mul":
            continue
        candidates = {"vLLM C": r.get("vllm_c_ms")}
        if "hf_ms" in r:
            candidates["HF"] = r["hf_ms"]
        if "helion_ms" in r:
            candidates["Helion"] = r["helion_ms"]
        best = min((k for k, v in candidates.items() if v is not None), key=lambda k: candidates[k])
        lines.append(
            "| {shape} | {torch:.4f} | {vllm:.4f} | {hf} | {helion} | {hvc} | {best} |".format(
                shape="x".join(str(x) for x in r["shape"]),
                torch=r["torch_ms"],
                vllm=r["vllm_c_ms"],
                hf=f"{r['hf_ms']:.4f}" if "hf_ms" in r else "n/a",
                helion=f"{r['helion_ms']:.4f}" if "helion_ms" in r else "n/a",
                hvc=f"{r['helion_speedup_vs_vllm_c']:.2f}x"
                if "helion_speedup_vs_vllm_c" in r
                else "n/a",
                best=best,
            )
        )
    lines.append("")
    lines.append("## Scale Plus Residual RMSNorm")
    lines.append("")
    lines.append("| shape | torch ms | scale+vLLM C ms | Helion fused ms | Helion vs scale+vLLM C | best |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for r in results:
        if r.get("section") != "deepseek_mla_scale_plus_post_attention_rmsnorm":
            continue
        candidates = {"scale+vLLM C": r.get("vllm_scale_then_c_ms")}
        if "helion_ms" in r:
            candidates["Helion fused"] = r["helion_ms"]
        best = min((k for k, v in candidates.items() if v is not None), key=lambda k: candidates[k])
        lines.append(
            "| {shape} | {torch:.4f} | {vllm:.4f} | {helion} | {hvc} | {best} |".format(
                shape="x".join(str(x) for x in r["shape"]),
                torch=r["torch_ms"],
                vllm=r["vllm_scale_then_c_ms"],
                helion=f"{r['helion_ms']:.4f}" if "helion_ms" in r else "n/a",
                hvc=f"{r['helion_speedup_vs_vllm_scale_then_c']:.2f}x"
                if "helion_speedup_vs_vllm_scale_then_c" in r
                else "n/a",
                best=best,
            )
        )
    lines.append("")
    lines.append("## MoE CuTeDSL/CUTLASS Probe")
    lines.append("")
    for r in results:
        if r.get("section") == "deepseek_moe_backend_probe":
            lines.append(f"- `{json.dumps(r, sort_keys=True)}`")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- `SiluAndMul` is already a vLLM C custom op in DeepSeek dense MLP layers; replace it only if a candidate beats that kernel, not just raw Torch.")
    lines.append("- `scale + residual RMSNorm` is a real DeepSeek MLA FP16 branch and is a plausible Helion fusion target because vLLM currently performs scaling separately before the fused RMSNorm.")
    lines.append("- FlashInfer CuTeDSL MoE is mainly an NVFP4/SM100-family path in this vLLM tree; on A10 it is expected to be unavailable.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--act-shapes",
        nargs="+",
        default=["128x2048", "1024x2048", "1024x4096", "8192x3072"],
    )
    parser.add_argument(
        "--norm-shapes",
        nargs="+",
        default=["128x3072", "1024x3072", "8192x3072"],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    results = []
    hf_activation = load_hf_activation()
    for shape in args.act_shapes:
        print(f"benchmarking DeepSeek SiluAndMul {shape}", flush=True)
        results.append(bench_silu_and_mul(shape, args.warmup, args.iters, hf_activation))
    for shape in args.norm_shapes:
        print(f"benchmarking DeepSeek scale+RMSNorm {shape}", flush=True)
        results.append(bench_scaled_add_norm(shape, args.warmup, args.iters))
    results.extend(probe_moe_backends())

    json_path = OUT / "deepseek_pipeline_fusions.json"
    md_path = OUT / "deepseek_pipeline_fusions.md"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    write_markdown(results, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
