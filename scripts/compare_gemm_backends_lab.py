import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import torch


ROOT = Path("/scratch/deepseek-prof")
CUTLASS_EXAMPLES = ROOT / "src" / "cutlass" / "examples" / "python" / "CuTeDSL"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(CUTLASS_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(CUTLASS_EXAMPLES))

from cutlass_gemm_lab import compile_static_bmm  # noqa: E402
from helion_gemm_bottleneck_lab import helion_matmul  # noqa: E402


OUT = ROOT / "profiles" / "gemm_backend_compare"
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


def abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().detach().cpu())


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = torch.maximum(a.abs(), b.abs()).clamp_min(1e-5)
    return float(((a - b).abs() / denom).max().detach().cpu())


def bench_shape(spec: str, warmup: int, iters: int) -> dict:
    m, k, n = [int(x) for x in spec.lower().split("x")]
    torch.manual_seed(0)
    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    b = torch.randn((k, n), dtype=torch.float16, device="cuda")

    torch_ms, torch_y = cuda_time_ms(lambda: a @ b, warmup, iters)

    helion_first_start = time.time()
    helion_first_y = helion_matmul(a, b)
    torch.cuda.synchronize()
    helion_first_s = time.time() - helion_first_start
    helion_ms, helion_y = cuda_time_ms(lambda: helion_matmul(a, b), warmup, iters)

    a_bmm = a.unsqueeze(0).contiguous()
    b_bmm = b.unsqueeze(0).contiguous()
    c_bmm = torch.empty((1, m, n), dtype=torch.float16, device="cuda")
    cutedsl_compile_start = time.time()
    cutedsl_bmm = compile_static_bmm(1, m, n, k)
    torch.cuda.synchronize()
    cutedsl_compile_s = time.time() - cutedsl_compile_start

    def run_cutedsl() -> torch.Tensor:
        cutedsl_bmm(a_bmm, b_bmm, c_bmm)
        return c_bmm.squeeze(0)

    cutedsl_first_start = time.time()
    cutedsl_first_y = run_cutedsl()
    torch.cuda.synchronize()
    cutedsl_first_s = time.time() - cutedsl_first_start
    cutedsl_ms, cutedsl_y = cuda_time_ms(run_cutedsl, warmup, iters)

    flops = 2 * m * n * k
    return {
        "shape_mkn": [m, k, n],
        "dtype": "torch.float16",
        "torch_mm_ms": torch_ms,
        "helion_mm_ms": helion_ms,
        "cutedsl_bmm_ms": cutedsl_ms,
        "helion_speedup_vs_torch": torch_ms / helion_ms if helion_ms else None,
        "cutedsl_speedup_vs_torch": torch_ms / cutedsl_ms if cutedsl_ms else None,
        "helion_first_call_s": helion_first_s,
        "cutedsl_compile_s": cutedsl_compile_s,
        "cutedsl_first_call_s": cutedsl_first_s,
        "torch_tflops": flops / (torch_ms * 1e-3) / 1e12,
        "helion_tflops": flops / (helion_ms * 1e-3) / 1e12,
        "cutedsl_tflops": flops / (cutedsl_ms * 1e-3) / 1e12,
        "helion_max_abs_error": abs_error(torch_y, helion_y),
        "helion_max_rel_error": rel_error(torch_y, helion_y),
        "cutedsl_max_abs_error": abs_error(torch_y, cutedsl_y),
        "cutedsl_max_rel_error": rel_error(torch_y, cutedsl_y),
        "helion_first_max_abs_error": abs_error(torch_y, helion_first_y),
        "cutedsl_first_max_abs_error": abs_error(torch_y, cutedsl_first_y),
    }


def write_markdown(results: list[dict], path: Path) -> None:
    lines = ["# GEMM Backend Compare", ""]
    lines.append("| shape `(M,K,N)` | torch mm | Helion | CuTeDSL | Helion speedup | CuTeDSL speedup | torch TFLOP/s | Helion TFLOP/s | CuTeDSL TFLOP/s |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        lines.append(
            "| ({}) | {:.4f} | {:.4f} | {:.4f} | {:.3f} | {:.3f} | {:.2f} | {:.2f} | {:.2f} |".format(
                ", ".join(str(x) for x in r["shape_mkn"]),
                r["torch_mm_ms"],
                r["helion_mm_ms"],
                r["cutedsl_bmm_ms"],
                r["helion_speedup_vs_torch"],
                r["cutedsl_speedup_vs_torch"],
                r["torch_tflops"],
                r["helion_tflops"],
                r["cutedsl_tflops"],
            )
        )
    lines.append("")
    lines.append("## Error Check")
    lines.append("")
    lines.append("| shape `(M,K,N)` | Helion abs | Helion rel | CuTeDSL abs | CuTeDSL rel |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        lines.append(
            "| ({}) | {:.3g} | {:.3g} | {:.3g} | {:.3g} |".format(
                ", ".join(str(x) for x in r["shape_mkn"]),
                r["helion_max_abs_error"],
                r["helion_max_rel_error"],
                r["cutedsl_max_abs_error"],
                r["cutedsl_max_rel_error"],
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- CuTeDSL is run through the CUTLASS Ampere `TensorOpGemm` BMM example with batch size `1`.")
    lines.append("- These are standalone proxy shapes for the current vLLM decode-linear bottleneck, not a vLLM integration.")
    lines.append("- For real serving impact, replace a traced vLLM op behind an env flag and re-run the realistic matrix.")
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
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if not CUTLASS_EXAMPLES.exists():
        raise SystemExit(f"CUTLASS examples clone missing: {CUTLASS_EXAMPLES}")

    results = [bench_shape(spec, args.warmup, args.iters) for spec in args.shapes]
    json_path = OUT / "gemm_backend_compare.json"
    md_path = OUT / "gemm_backend_compare.md"
    json_path.write_text(json.dumps(results, indent=2))
    write_markdown(results, md_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
