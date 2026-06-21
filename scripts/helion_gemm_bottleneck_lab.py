import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch

import helion
import helion.language as hl


ROOT = Path("/scratch/deepseek-prof")
OUT = ROOT / "profiles" / "helion_gemm"
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


def abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().detach().cpu())


@helion.kernel(
    static_shapes=True,
    autotune_effort="quick",
    autotune_budget_seconds=60,
    autotune_log=str(OUT / "matmul_autotune"),
    autotune_baseline_fn=lambda a, b: a @ b,
    autotune_baseline_atol=1e-1,
    autotune_baseline_rtol=1e-1,
)
def helion_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = a.size(0)
    k = a.size(1)
    n = b.size(1)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc

    return out


def bench_shape(spec: str, warmup: int, iters: int) -> dict:
    m, k, n = [int(x) for x in spec.lower().split("x")]
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)

    result = {
        "shape_mkn": [m, k, n],
        "dtype": "torch.float16",
    }

    compile_started = time.time()
    helion_first_y = helion_matmul(a, b)
    torch.cuda.synchronize()
    result["helion_first_call_s"] = time.time() - compile_started

    torch_ms, torch_y = cuda_time_ms(lambda: a @ b, warmup, iters)
    helion_ms, helion_y = cuda_time_ms(lambda: helion_matmul(a, b), warmup, iters)

    flops = 2 * m * n * k
    result.update(
        {
            "torch_mm_ms": torch_ms,
            "helion_mm_ms": helion_ms,
            "helion_speedup_vs_torch_mm": torch_ms / helion_ms if helion_ms else None,
            "torch_tflops": flops / (torch_ms * 1e-3) / 1e12,
            "helion_tflops": flops / (helion_ms * 1e-3) / 1e12,
            "max_rel_error_first": rel_error(torch_y, helion_first_y),
            "max_rel_error_after_timing": rel_error(torch_y, helion_y),
            "max_abs_error_after_timing": abs_error(torch_y, helion_y),
        }
    )
    return result


def write_markdown(results: list[dict], path: Path) -> None:
    lines = ["# Helion GEMM Bottleneck Lab", ""]
    lines.append(
        "| shape `(M,K,N)` | torch mm | Helion mm | speedup | torch TFLOP/s | Helion TFLOP/s | first call s | max abs err | max rel err |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        lines.append(
            "| ({}) | {:.4f} | {:.4f} | {:.3f} | {:.2f} | {:.2f} | {:.2f} | {:.3g} | {:.3g} |".format(
                ", ".join(str(x) for x in r["shape_mkn"]),
                r["torch_mm_ms"],
                r["helion_mm_ms"],
                r["helion_speedup_vs_torch_mm"],
                r["torch_tflops"],
                r["helion_tflops"],
                r["helion_first_call_s"],
                r["max_abs_error_after_timing"],
                r["max_rel_error_after_timing"],
            )
        )
    lines.append("")
    lines.append("## Shape Notes")
    lines.append("")
    lines.append("- These are decode-linear proxy shapes, not a vLLM integration yet.")
    lines.append("- `M` is the active token/request batch dimension; small `M` is the difficult decode case.")
    lines.append("- `K=768,N=3072` and `K=3072,N=768` mimic OPT-125M MLP projections from the current local realistic workload.")
    lines.append("- `K=768,N=768` mimics attention/output projections.")
    lines.append("- A serving win requires replacing the actual vLLM path and re-running the realistic matrix.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--shapes",
        nargs="*",
        default=[
            "8x768x3072",
            "16x768x3072",
            "32x768x3072",
            "8x3072x768",
            "16x3072x768",
            "32x3072x768",
            "8x768x768",
            "16x768x768",
            "32x768x768",
        ],
        help="Shape format: MxKxN for A[M,K] @ B[K,N]",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    results = [bench_shape(spec, args.warmup, args.iters) for spec in args.shapes]
    json_path = OUT / "helion_gemm_bottleneck_lab.json"
    md_path = OUT / "helion_gemm_bottleneck_lab.md"
    json_path.write_text(json.dumps(results, indent=2))
    write_markdown(results, md_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
