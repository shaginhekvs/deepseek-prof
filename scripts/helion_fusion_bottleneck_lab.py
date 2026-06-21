import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import helion
import helion.language as hl


ROOT = Path("/scratch/deepseek-prof")
OUT = ROOT / "profiles" / "helion_fusion"
OUT.mkdir(parents=True, exist_ok=True)
AUTOTUNE_LOG_PREFIXES = [
    "silu_mul_residual_autotune",
    "add_rms_norm_autotune",
    "silu_mul_add_rms_norm_autotune",
]


def cuda_time_ms(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> tuple[float, torch.Tensor]:
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


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = torch.maximum(a.abs(), b.abs()).clamp_min(1e-5)
    return float(((a - b).abs() / denom).max().detach().cpu())


def abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().detach().cpu())


def torch_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * w


def torch_silu_mul_residual(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return F.silu(a) * b + residual


def torch_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return torch_rms_norm(x + residual, w, eps)


def torch_silu_mul_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float
) -> torch.Tensor:
    return torch_rms_norm(torch_silu_mul_residual(x, residual), w, eps)


@helion.kernel(
    static_shapes=True,
    autotune_effort="quick",
    autotune_budget_seconds=45,
    autotune_log=str(OUT / "silu_mul_residual_autotune"),
    autotune_baseline_fn=torch_silu_mul_residual,
    autotune_baseline_atol=1e-2,
    autotune_baseline_rtol=1e-2,
)
def helion_silu_mul_residual(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    rows = x.size(0)
    hidden = residual.size(1)
    out = torch.empty_like(residual)

    for tile_m, tile_n in hl.tile([rows, hidden]):
        a = x[tile_m, tile_n]
        b = x[tile_m, tile_n + hidden]
        out[tile_m, tile_n] = a * torch.sigmoid(a) * b + residual[tile_m, tile_n]

    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="quick",
    autotune_budget_seconds=45,
    autotune_log=str(OUT / "add_rms_norm_autotune"),
    autotune_baseline_fn=torch_add_rms_norm,
    autotune_baseline_atol=1e-2,
    autotune_baseline_rtol=1e-2,
)
def helion_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float
) -> torch.Tensor:
    rows = x.size(0)
    out = torch.empty_like(x)

    for tile_m in hl.tile(rows):
        y = x[tile_m, :] + residual[tile_m, :]
        variance = (y * y).mean(dim=-1, keepdim=True)
        out[tile_m, :] = y * torch.rsqrt(variance + eps) * w[None, :]

    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="quick",
    autotune_budget_seconds=45,
    autotune_log=str(OUT / "silu_mul_add_rms_norm_autotune"),
    autotune_baseline_fn=torch_silu_mul_add_rms_norm,
    autotune_baseline_atol=1e-2,
    autotune_baseline_rtol=1e-2,
)
def helion_silu_mul_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float
) -> torch.Tensor:
    rows = x.size(0)
    hidden = residual.size(1)
    out = torch.empty_like(residual)

    for tile_m in hl.tile(rows):
        a = x[tile_m, :hidden]
        b = x[tile_m, hidden:]
        y = a * torch.sigmoid(a) * b + residual[tile_m, :]
        variance = (y * y).mean(dim=-1, keepdim=True)
        out[tile_m, :] = y * torch.rsqrt(variance + eps) * w[None, :]

    return out


def load_hf_activation():
    try:
        from kernels import get_kernel

        return get_kernel("kernels-community/activation", version=1, trust_remote_code=True)
    except Exception:
        return None


def bench_one(name: str, baseline_fn: Callable[[], torch.Tensor], helion_fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> dict:
    base_ms, base_y = cuda_time_ms(baseline_fn, warmup, iters)
    result = {"op": name, "baseline_ms": base_ms}
    try:
        compile_start = time.time()
        helion_first_y = helion_fn()
        torch.cuda.synchronize()
        first_call_s = time.time() - compile_start

        helion_ms, helion_y = cuda_time_ms(helion_fn, warmup, iters)
        result.update(
            {
                "status": "ok",
                "helion_ms": helion_ms,
                "helion_speedup_vs_baseline": base_ms / helion_ms if helion_ms else None,
                "helion_first_call_s": first_call_s,
                "helion_max_abs_error": abs_error(base_y, helion_y),
                "helion_max_rel_error": rel_error(base_y, helion_y),
                "helion_first_max_abs_error": abs_error(base_y, helion_first_y),
            }
        )
    except Exception as exc:
        result.update({"status": "helion_error", "error": repr(exc)})
    return result


def bench_shape(spec: str, warmup: int, iters: int) -> list[dict]:
    rows, hidden = [int(x) for x in spec.lower().split("x")]
    dtype = torch.float16
    eps = 1e-5
    torch.manual_seed(0)
    x_gate = torch.randn((rows, hidden * 2), device="cuda", dtype=dtype)
    x = torch.randn((rows, hidden), device="cuda", dtype=dtype)
    residual = torch.randn((rows, hidden), device="cuda", dtype=dtype)
    w = torch.randn((hidden,), device="cuda", dtype=dtype)
    hf_activation = load_hf_activation()

    rows_out = []
    row_base = {"shape": [rows, hidden], "dtype": str(dtype)}

    rows_out.append(
        row_base
        | bench_one(
            "silu_mul_residual",
            lambda: torch_silu_mul_residual(x_gate, residual),
            lambda: helion_silu_mul_residual(x_gate, residual),
            warmup,
            iters,
        )
    )

    if hf_activation is not None:
        tmp = torch.empty_like(residual)

        def hf_silu_mul_residual() -> torch.Tensor:
            hf_activation.silu_and_mul(tmp, x_gate)
            return tmp + residual

        hf_ms, hf_y = cuda_time_ms(hf_silu_mul_residual, warmup, iters)
        rows_out[-1]["hf_activation_plus_add_ms"] = hf_ms
        rows_out[-1]["hf_activation_plus_add_speedup_vs_baseline"] = rows_out[-1]["baseline_ms"] / hf_ms
        rows_out[-1]["hf_activation_plus_add_max_abs_error"] = abs_error(
            torch_silu_mul_residual(x_gate, residual), hf_y
        )

    rows_out.append(
        row_base
        | bench_one(
            "add_rms_norm",
            lambda: torch_add_rms_norm(x, residual, w, eps),
            lambda: helion_add_rms_norm(x, residual, w, eps),
            warmup,
            iters,
        )
    )

    rows_out.append(
        row_base
        | bench_one(
            "silu_mul_add_rms_norm",
            lambda: torch_silu_mul_add_rms_norm(x_gate, residual, w, eps),
            lambda: helion_silu_mul_add_rms_norm(x_gate, residual, w, eps),
            warmup,
            iters,
        )
    )

    return rows_out


def snapshot_autotune_logs(spec: str) -> None:
    safe_spec = spec.lower().replace("x", "_")
    for prefix in AUTOTUNE_LOG_PREFIXES:
        for suffix in ["csv", "log"]:
            src = OUT / f"{prefix}.{suffix}"
            if src.exists():
                dst = OUT / f"{safe_spec}_{prefix}.{suffix}"
                shutil.copy2(src, dst)


def write_markdown(results: list[dict], path: Path) -> None:
    lines = ["# Helion Fusion Bottleneck Lab", ""]
    lines.append("| op | shape | baseline ms | Helion ms | speedup | first call s | max abs err | max rel err | HF+add ms |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        if r.get("status") == "helion_error":
            lines.append(
                "| {op} | ({shape0}, {shape1}) | {baseline_ms:.4f} |  |  |  |  |  |  |".format(
                    op=r["op"],
                    shape0=r["shape"][0],
                    shape1=r["shape"][1],
                    baseline_ms=r["baseline_ms"],
                )
            )
            continue
        lines.append(
            "| {op} | ({shape0}, {shape1}) | {baseline_ms:.4f} | {helion_ms:.4f} | {helion_speedup_vs_baseline:.3f} | {helion_first_call_s:.2f} | {helion_max_abs_error:.3g} | {helion_max_rel_error:.3g} | {hf_ms} |".format(
                op=r["op"],
                shape0=r["shape"][0],
                shape1=r["shape"][1],
                baseline_ms=r["baseline_ms"],
                helion_ms=r["helion_ms"],
                helion_speedup_vs_baseline=r["helion_speedup_vs_baseline"],
                helion_first_call_s=r["helion_first_call_s"],
                helion_max_abs_error=r["helion_max_abs_error"],
                helion_max_rel_error=r["helion_max_rel_error"],
                hf_ms=f"{r['hf_activation_plus_add_ms']:.4f}" if "hf_activation_plus_add_ms" in r else "",
            )
        )
    lines.append("")
    failures = [r for r in results if r.get("status") == "helion_error"]
    if failures:
        lines.append("## Helion Failures")
        lines.append("")
        for r in failures:
            lines.append(f"- `{r['op']}` shape `{tuple(r['shape'])}`: `{r['error']}`")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- These are fusion candidates around the current GEMM/MLP/norm bottleneck area, not replacements for GEMM itself.")
    lines.append("- `silu_mul_residual` tests whether Helion can beat separate activation/gate plus residual add.")
    lines.append("- `add_rms_norm` tests a common residual-plus-normalization fusion.")
    lines.append("- `silu_mul_add_rms_norm` tests a larger fused chain that removes multiple intermediate writes.")
    lines.append("- A vLLM integration should only follow for a row where Helion beats both raw Torch and the existing library path.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--shapes", nargs="*", default=["1024x768", "8192x768", "1024x3072", "8192x3072"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    results: list[dict] = []
    for spec in args.shapes:
        results.extend(bench_shape(spec, args.warmup, args.iters))
        if os.environ.get("HELION_SNAPSHOT_AUTOTUNE") == "1":
            snapshot_autotune_logs(spec)

    json_path = OUT / "helion_fusion_bottleneck_lab.json"
    md_path = OUT / "helion_fusion_bottleneck_lab.md"
    json_path.write_text(json.dumps(results, indent=2))
    write_markdown(results, md_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
