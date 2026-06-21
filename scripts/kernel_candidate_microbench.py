import argparse
import importlib
import json
import math
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


ROOT = Path("/scratch/deepseek-prof")
OUT = ROOT / "profiles" / "kernel_candidates"
OUT.mkdir(parents=True, exist_ok=True)


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


def torch_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * w


def torch_swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.silu(a) * b


def bench_liger(results: list[dict], warmup: int, iters: int) -> None:
    try:
        from liger_kernel.transformers import functional as LF
    except Exception as exc:
        results.append({"library": "liger", "status": "import_error", "error": repr(exc)})
        return

    dtype = torch.float16
    shapes = [(1024, 768), (8192, 768), (1024, 3072), (8192, 3072)]
    eps = 1e-5

    for shape in shapes:
        x = torch.randn(shape, device="cuda", dtype=dtype)
        w = torch.randn((shape[-1],), device="cuda", dtype=dtype)
        b = torch.randn((shape[-1],), device="cuda", dtype=dtype)

        base_ms, base_y = cuda_time_ms(lambda: torch_rms_norm(x, w, eps), warmup, iters)
        liger_ms, liger_y = cuda_time_ms(
            lambda: LF.liger_rms_norm(x.clone(), w, eps, in_place=True),
            warmup,
            iters,
        )
        results.append(
            {
                "library": "liger",
                "op": "rms_norm",
                "shape": shape,
                "baseline_ms": base_ms,
                "candidate_ms": liger_ms,
                "speedup": base_ms / liger_ms if liger_ms else None,
                "max_rel_error": rel_error(base_y, liger_y),
            }
        )

        base_ms, base_y = cuda_time_ms(lambda: F.layer_norm(x, (shape[-1],), w, b, eps), warmup, iters)
        liger_ms, liger_y = cuda_time_ms(lambda: LF.liger_layer_norm(x, w, b, eps), warmup, iters)
        results.append(
            {
                "library": "liger",
                "op": "layer_norm",
                "shape": shape,
                "baseline_ms": base_ms,
                "candidate_ms": liger_ms,
                "speedup": base_ms / liger_ms if liger_ms else None,
                "max_rel_error": rel_error(base_y, liger_y),
            }
        )

    for shape in [(1024, 3072), (8192, 3072)]:
        a = torch.randn(shape, device="cuda", dtype=dtype)
        b = torch.randn(shape, device="cuda", dtype=dtype)
        base_ms, base_y = cuda_time_ms(lambda: torch_swiglu(a, b), warmup, iters)
        liger_ms, liger_y = cuda_time_ms(lambda: LF.liger_swiglu(a, b), warmup, iters)
        results.append(
            {
                "library": "liger",
                "op": "swiglu",
                "shape": shape,
                "baseline_ms": base_ms,
                "candidate_ms": liger_ms,
                "speedup": base_ms / liger_ms if liger_ms else None,
                "max_rel_error": rel_error(base_y, liger_y),
            }
        )

    for shape in [(1024, 50272), (128, 50272)]:
        x = torch.randn(shape, device="cuda", dtype=dtype)
        base_ms, base_y = cuda_time_ms(lambda: torch.softmax(x, dim=-1), warmup, iters)
        liger_ms, liger_y = cuda_time_ms(lambda: LF.liger_softmax(x), warmup, iters)
        results.append(
            {
                "library": "liger",
                "op": "softmax",
                "shape": shape,
                "baseline_ms": base_ms,
                "candidate_ms": liger_ms,
                "speedup": base_ms / liger_ms if liger_ms else None,
                "max_rel_error": rel_error(base_y, liger_y),
            }
        )


def bench_hf_kernels(results: list[dict], warmup: int, iters: int) -> None:
    try:
        from kernels import get_kernel
    except Exception as exc:
        results.append({"library": "hf_kernels", "status": "import_error", "error": repr(exc)})
        return

    repos = [
        "kernels-community/activation",
        "kernels-community/triton_kernels",
    ]
    activation = None
    for repo in repos:
        try:
            kernel = get_kernel(repo, version=1, trust_remote_code=True)
            if repo == "kernels-community/activation":
                activation = kernel
            results.append(
                {
                    "library": "hf_kernels",
                    "repo": repo,
                    "status": "loaded",
                    "attrs": [a for a in dir(kernel) if not a.startswith("_")][:80],
                }
            )
        except Exception as exc:
            results.append(
                {
                    "library": "hf_kernels",
                    "repo": repo,
                    "status": "load_error",
                    "error": repr(exc),
                }
            )

    if activation is not None:
        dtype = torch.float16
        for rows, hidden in [(1024, 3072), (8192, 3072)]:
            x = torch.randn((rows, hidden * 2), device="cuda", dtype=dtype)
            a, b = x.chunk(2, dim=-1)
            out = torch.empty((rows, hidden), device="cuda", dtype=dtype)
            base_ms, base_y = cuda_time_ms(lambda: F.silu(a) * b, warmup, iters)

            def hf_silu_and_mul() -> torch.Tensor:
                activation.silu_and_mul(out, x)
                return out

            hf_ms, hf_y = cuda_time_ms(hf_silu_and_mul, warmup, iters)
            results.append(
                {
                    "library": "hf_kernels",
                    "repo": "kernels-community/activation",
                    "op": "silu_and_mul",
                    "shape": (rows, hidden),
                    "baseline_ms": base_ms,
                    "candidate_ms": hf_ms,
                    "speedup": base_ms / hf_ms if hf_ms else None,
                    "max_rel_error": rel_error(base_y, hf_y),
                }
            )

            base_ms, base_y = cuda_time_ms(lambda: F.gelu(a, approximate="tanh") * b, warmup, iters)

            def hf_gelu_tanh_and_mul() -> torch.Tensor:
                activation.gelu_tanh_and_mul(out, x)
                return out

            hf_ms, hf_y = cuda_time_ms(hf_gelu_tanh_and_mul, warmup, iters)
            results.append(
                {
                    "library": "hf_kernels",
                    "repo": "kernels-community/activation",
                    "op": "gelu_tanh_and_mul",
                    "shape": (rows, hidden),
                    "baseline_ms": base_ms,
                    "candidate_ms": hf_ms,
                    "speedup": base_ms / hf_ms if hf_ms else None,
                    "max_rel_error": rel_error(base_y, hf_y),
                }
            )


def bench_quack(results: list[dict], warmup: int, iters: int) -> None:
    try:
        import quack
    except Exception as exc:
        results.append({"library": "quack", "status": "import_error", "error": repr(exc)})
        return

    dtype = torch.float16
    eps = 1e-6
    for shape in [(1024, 768), (8192, 768), (1024, 3072), (8192, 3072)]:
        x = torch.randn(shape, device="cuda", dtype=dtype)
        w = torch.randn((shape[-1],), device="cuda", dtype=dtype)
        base_ms, base_y = cuda_time_ms(lambda: torch_rms_norm(x, w, eps), warmup, iters)
        quack_ms, quack_y = cuda_time_ms(lambda: quack.rmsnorm(x, weight=w, eps=eps), warmup, iters)
        results.append(
            {
                "library": "quack",
                "op": "rms_norm",
                "shape": shape,
                "baseline_ms": base_ms,
                "candidate_ms": quack_ms,
                "speedup": base_ms / quack_ms if quack_ms else None,
                "max_rel_error": rel_error(base_y, quack_y),
            }
        )

    for shape in [(1024, 50272), (128, 50272)]:
        x = torch.randn(shape, device="cuda", dtype=dtype)
        base_ms, base_y = cuda_time_ms(lambda: torch.softmax(x, dim=-1), warmup, iters)
        try:
            quack_ms, quack_y = cuda_time_ms(lambda: quack.softmax(x), warmup, iters)
            results.append(
                {
                    "library": "quack",
                    "op": "softmax",
                    "shape": shape,
                    "baseline_ms": base_ms,
                    "candidate_ms": quack_ms,
                    "speedup": base_ms / quack_ms if quack_ms else None,
                    "max_rel_error": rel_error(base_y, quack_y),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "library": "quack",
                    "op": "softmax",
                    "shape": shape,
                    "status": "bench_error",
                    "baseline_ms": base_ms,
                    "error": repr(exc),
                }
            )


def inspect_existing(results: list[dict]) -> None:
    for name in ["humming", "quack", "tokenspeed_mla", "tokenspeed_triton", "cutlass"]:
        try:
            mod = importlib.import_module(name)
            results.append(
                {
                    "library": name,
                    "status": "imported",
                    "file": getattr(mod, "__file__", None),
                    "attrs": [a for a in dir(mod) if not a.startswith("_")][:60],
                }
            )
        except Exception as exc:
            results.append({"library": name, "status": "import_error", "error": repr(exc)})


def write_markdown(results: list[dict], path: Path) -> None:
    bench_rows = [r for r in results if "candidate_ms" in r]
    failed_rows = [r for r in results if r.get("status") == "bench_error"]
    lines = ["# Kernel Candidate Microbench", ""]
    if bench_rows:
        lines.append("| library | op | shape | baseline ms | candidate ms | speedup | max rel err |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for r in bench_rows:
            lines.append(
                "| {library} | {op} | {shape} | {baseline_ms:.4f} | {candidate_ms:.4f} | {speedup:.3f} | {max_rel_error:.3g} |".format(
                    **r
                )
            )
        lines.append("")

    if failed_rows:
        lines.append("## Bench Failures")
        for r in failed_rows:
            lines.append(f"- `{r.get('library')}` `{r.get('op')}` {r.get('shape')}: `{r.get('error')}`")
        lines.append("")

    lines.append("## Import/Load Status")
    for r in results:
        if "baseline_ms" not in r and r.get("status") != "bench_error":
            lines.append(f"- `{r.get('library')}` {r.get('repo', '')}: {r.get('status', 'ok')}")
            if "error" in r:
                lines.append(f"  - error: `{r['error']}`")
            if "attrs" in r:
                lines.append(f"  - attrs: `{', '.join(r['attrs'][:20])}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    torch.manual_seed(0)
    results: list[dict] = []
    inspect_existing(results)
    bench_liger(results, args.warmup, args.iters)
    bench_hf_kernels(results, args.warmup, args.iters)
    bench_quack(results, args.warmup, args.iters)

    json_path = OUT / "kernel_candidate_microbench.json"
    md_path = OUT / "kernel_candidate_microbench.md"
    json_path.write_text(json.dumps(results, indent=2))
    write_markdown(results, md_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
