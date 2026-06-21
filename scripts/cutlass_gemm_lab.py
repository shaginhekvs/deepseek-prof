# Adapted for local benchmarking from NVIDIA CUTLASS CuTe DSL Ampere examples.
# Upstream source: https://github.com/NVIDIA/cutlass

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import torch

import cutlass
import cutlass.cute as cute


ROOT = Path("/scratch/deepseek-prof")
CUTLASS_EXAMPLES = ROOT / "src" / "cutlass" / "examples" / "python" / "CuTeDSL"
if str(CUTLASS_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(CUTLASS_EXAMPLES))

from cute.ampere.kernel.dense_gemm.tensorop_gemm import TensorOpGemm  # noqa: E402


OUT = ROOT / "profiles" / "cutlass"
OUT.mkdir(parents=True, exist_ok=True)


@cute.jit
def cutlass_bmm(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
):
    gemm_op = TensorOpGemm(cutlass.Float16, cutlass.Float16, cutlass.Float32, (2, 2, 1))

    # TensorOpGemm expects CuTe convention: A=(m,k,l), B=(n,k,l), C=(m,n,l).
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))

    gemm_op(a, b, c)


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


def compile_static_bmm(batch: int, m: int, n: int, k: int):
    from cutlass.cute.runtime import make_fake_compact_tensor

    fake_a = make_fake_compact_tensor(
        cutlass.Float16, (batch, m, k), stride_order=(2, 1, 0), assumed_align=16
    )
    fake_b = make_fake_compact_tensor(
        cutlass.Float16, (batch, k, n), stride_order=(2, 1, 0), assumed_align=16
    )
    fake_c = make_fake_compact_tensor(
        cutlass.Float16, (batch, m, n), stride_order=(2, 1, 0), assumed_align=16
    )
    return cute.compile(cutlass_bmm, fake_a, fake_b, fake_c, options="--enable-tvm-ffi")


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = torch.maximum(a.abs(), b.abs()).clamp_min(1e-5)
    return float(((a - b).abs() / denom).max().detach().cpu())


def bench_shape(spec: str, warmup: int, iters: int) -> dict:
    batch, m, n, k = [int(x) for x in spec.lower().split("x")]
    torch.manual_seed(0)
    a = torch.randn(batch, m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(batch, k, n, dtype=torch.float16, device="cuda")
    c = torch.empty(batch, m, n, dtype=torch.float16, device="cuda")

    compile_start = time.time()
    compiled = compile_static_bmm(batch, m, n, k)
    torch.cuda.synchronize()
    compile_s = time.time() - compile_start

    def run_cutlass() -> torch.Tensor:
        compiled(a, b, c)
        return c

    # First call makes any runtime setup visible separately from steady state.
    first_start = time.time()
    first_y = run_cutlass()
    torch.cuda.synchronize()
    first_call_s = time.time() - first_start

    ref = torch.bmm(a, b)
    torch.testing.assert_close(first_y, ref, atol=1e-1, rtol=1e-1)

    torch_ms, torch_y = cuda_time_ms(lambda: torch.bmm(a, b), warmup, iters)
    cutlass_ms, cutlass_y = cuda_time_ms(run_cutlass, warmup, iters)

    flops = 2 * batch * m * n * k
    return {
        "shape": [batch, m, n, k],
        "dtype": "torch.float16",
        "torch_bmm_ms": torch_ms,
        "cutlass_bmm_ms": cutlass_ms,
        "cutlass_speedup_vs_torch_bmm": torch_ms / cutlass_ms if cutlass_ms else None,
        "cutlass_compile_s": compile_s,
        "cutlass_first_call_s": first_call_s,
        "torch_tflops": flops / (torch_ms * 1e-3) / 1e12,
        "cutlass_tflops": flops / (cutlass_ms * 1e-3) / 1e12,
        "max_rel_error": rel_error(torch_y, cutlass_y),
    }


def write_markdown(results: list[dict], path: Path) -> None:
    lines = ["# CUTLASS GEMM Lab", ""]
    lines.append("| shape `(B,M,N,K)` | torch.bmm ms | CUTLASS ms | speedup | torch TFLOP/s | CUTLASS TFLOP/s | compile s | first call s | max rel err |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        lines.append(
            "| ({}) | {:.4f} | {:.4f} | {:.3f} | {:.2f} | {:.2f} | {:.2f} | {:.4f} | {:.3g} |".format(
                ", ".join(str(x) for x in r["shape"]),
                r["torch_bmm_ms"],
                r["cutlass_bmm_ms"],
                r["cutlass_speedup_vs_torch_bmm"],
                r["torch_tflops"],
                r["cutlass_tflops"],
                r["cutlass_compile_s"],
                r["cutlass_first_call_s"],
                r["max_rel_error"],
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This uses NVIDIA CUTLASS CuTe DSL `TensorOpGemm` from the cloned upstream examples.")
    lines.append("- `torch.bmm` is usually cuBLAS-backed, so this comparison is CUTLASS vs a strong vendor library baseline.")
    lines.append("- The current kernel is a plain fp16 BMM. For vLLM wins, the more interesting CUTLASS work is fused epilogues or shape-specialized small decode GEMMs.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--shapes",
        nargs="*",
        default=["2x512x512x256", "8x128x128x512", "32x64x64x512"],
        help="Shape format: BxMxNxK for A[B,M,K] @ B[B,K,N]",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if not CUTLASS_EXAMPLES.exists():
        raise SystemExit(f"CUTLASS examples clone missing: {CUTLASS_EXAMPLES}")

    results = [bench_shape(spec, args.warmup, args.iters) for spec in args.shapes]
    json_path = OUT / "cutlass_gemm_lab.json"
    md_path = OUT / "cutlass_gemm_lab.md"
    json_path.write_text(json.dumps(results, indent=2))
    write_markdown(results, md_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
