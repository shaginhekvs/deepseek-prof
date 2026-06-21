#!/usr/bin/env python3
"""Benchmark vLLM's DeepSeek-relevant fused_add_rms_norm providers."""

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch

import vllm.kernels  # noqa: F401
from vllm import ir
from vllm.ir.op import enable_torch_wrap
from vllm.kernels import helion_ops, vllm_c


ROOT = Path("/scratch/deepseek-prof")
OUT = ROOT / "profiles" / "helion_vllm_ir"
OUT.mkdir(parents=True, exist_ok=True)


def cuda_time_ms(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor]], warmup: int, iters: int
) -> tuple[float, tuple[torch.Tensor, torch.Tensor]]:
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
    assert y is not None
    return start.elapsed_time(end) / iters, y


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().detach().cpu())


def max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float()
    bf = b.float()
    denom = torch.maximum(af.abs(), bf.abs()).clamp_min(1e-5)
    return float(((af - bf).abs() / denom).max().detach().cpu())


def native_impl(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    with enable_torch_wrap(False):
        with ir.ops.fused_add_rms_norm.set_priority(["native"]):
            return ir.ops.fused_add_rms_norm.maybe_inplace(x, residual, weight, eps)


def vllm_c_impl(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return vllm_c.fused_add_rms_norm.impl_fn(x, residual, weight, eps)


def helion_impl(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return helion_ops.fused_add_rms_norm.impl_fn(x, residual, weight, eps)


def bench_shape(rows: int, hidden: int, warmup: int, iters: int) -> dict:
    dtype = torch.float16
    eps = 1e-5
    torch.manual_seed(rows * 100000 + hidden)
    x = torch.randn((rows, hidden), device="cuda", dtype=dtype)
    residual = torch.randn((rows, hidden), device="cuda", dtype=dtype)
    weight = torch.randn((hidden,), device="cuda", dtype=dtype)

    native_ms, native_y = cuda_time_ms(
        lambda: native_impl(x, residual, weight, eps), warmup, iters
    )
    c_x = x.clone()
    c_residual = residual.clone()
    c_ms, _ = cuda_time_ms(
        lambda: vllm_c_impl(c_x, c_residual, weight, eps), warmup, iters
    )
    c_y = vllm_c_impl(x.clone(), residual.clone(), weight, eps)

    row = {
        "shape": [rows, hidden],
        "dtype": str(dtype),
        "native_ms": native_ms,
        "vllm_c_ms": c_ms,
        "vllm_c_speedup_vs_native": native_ms / c_ms if c_ms else None,
        "vllm_c_out_abs_error": max_abs(native_y[0], c_y[0]),
        "vllm_c_residual_abs_error": max_abs(native_y[1], c_y[1]),
    }

    supported = helion_ops._helion_fused_add_rms_norm_supported(
        x, residual, weight, eps
    )
    row["helion_supported"] = supported
    if supported:
        compile_start = time.time()
        first_y = helion_impl(x, residual, weight, eps)
        torch.cuda.synchronize()
        first_call_s = time.time() - compile_start
        h_ms, h_y = cuda_time_ms(
            lambda: helion_impl(x, residual, weight, eps), warmup, iters
        )
        row.update(
            {
                "helion_ms": h_ms,
                "helion_first_call_s": first_call_s,
                "helion_speedup_vs_native": native_ms / h_ms if h_ms else None,
                "helion_speedup_vs_vllm_c": c_ms / h_ms if h_ms else None,
                "helion_out_abs_error": max_abs(native_y[0], h_y[0]),
                "helion_out_rel_error": max_rel(native_y[0], h_y[0]),
                "helion_residual_abs_error": max_abs(native_y[1], h_y[1]),
                "helion_first_out_abs_error": max_abs(native_y[0], first_y[0]),
            }
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["128x576", "1024x576", "128x3072", "1024x3072", "8192x3072"],
        help="rowsxhidden shapes for DeepSeek MLA-rank and hidden-size norms.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    rows = []
    for spec in args.shapes:
        m, n = [int(part) for part in spec.lower().split("x")]
        print(f"benchmarking fused_add_rms_norm {m}x{n}", flush=True)
        rows.append(bench_shape(m, n, args.warmup, args.iters))

    json_path = OUT / "vllm_deepseek_helion_rmsnorm.json"
    md_path = OUT / "vllm_deepseek_helion_rmsnorm.md"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        "# vLLM DeepSeek Helion RMSNorm Experiment",
        "",
        "This benchmarks vLLM's actual `fused_add_rms_norm` IR provider used by DeepSeek decoder residual RMSNorm.",
        "",
        "| shape | native ms | vllm_c ms | helion ms | helion vs native | helion vs vllm_c | first call s | out abs err | residual abs err |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        shape = "x".join(str(x) for x in row["shape"])
        lines.append(
            "| {shape} | {native:.4f} | {c:.4f} | {h} | {hn} | {hc} | {first} | {err} | {rerr} |".format(
                shape=shape,
                native=row["native_ms"],
                c=row["vllm_c_ms"],
                h=f"{row['helion_ms']:.4f}" if "helion_ms" in row else "n/a",
                hn=f"{row['helion_speedup_vs_native']:.2f}x"
                if "helion_speedup_vs_native" in row
                else "n/a",
                hc=f"{row['helion_speedup_vs_vllm_c']:.2f}x"
                if "helion_speedup_vs_vllm_c" in row
                else "n/a",
                first=f"{row['helion_first_call_s']:.2f}"
                if "helion_first_call_s" in row
                else "n/a",
                err=f"{row.get('helion_out_abs_error', 0):.4g}"
                if "helion_out_abs_error" in row
                else "n/a",
                rerr=f"{row.get('helion_residual_abs_error', 0):.4g}"
                if "helion_residual_abs_error" in row
                else "n/a",
            )
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
