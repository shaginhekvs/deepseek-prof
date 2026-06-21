import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import helion
import helion.language as hl


ROOT = Path("/scratch/deepseek-prof")
OUT = ROOT / "profiles" / "helion"
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


@helion.kernel(
    static_shapes=True,
    autotune_effort="quick",
    autotune_budget_seconds=45,
    autotune_log=str(OUT / "silu_and_mul_autotune"),
    autotune_baseline_fn=lambda x: F.silu(x[:, : x.size(1) // 2]) * x[:, x.size(1) // 2 :],
    autotune_baseline_atol=1e-2,
    autotune_baseline_rtol=1e-2,
)
def helion_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    rows = x.size(0)
    hidden = x.size(1) // 2
    out = torch.empty((rows, hidden), dtype=x.dtype, device=x.device)

    for tile_m, tile_n in hl.tile([rows, hidden]):
        a = x[tile_m, tile_n]
        b = x[tile_m, tile_n + hidden]
        out[tile_m, tile_n] = a * torch.sigmoid(a) * b

    return out


def load_hf_activation():
    try:
        from kernels import get_kernel

        return get_kernel("kernels-community/activation", version=1, trust_remote_code=True)
    except Exception:
        return None


def bench_shape(rows: int, hidden: int, warmup: int, iters: int) -> dict:
    dtype = torch.float16
    x = torch.randn((rows, hidden * 2), device="cuda", dtype=dtype)
    a, b = x.chunk(2, dim=-1)
    out = torch.empty((rows, hidden), device="cuda", dtype=dtype)
    hf_activation = load_hf_activation()

    result = {
        "shape": [rows, hidden],
        "dtype": str(dtype),
    }

    base_ms, base_y = cuda_time_ms(lambda: F.silu(a) * b, warmup, iters)
    result["torch_ms"] = base_ms

    compile_started = time.time()
    helion_y = helion_silu_and_mul(x)
    torch.cuda.synchronize()
    result["helion_first_call_s"] = time.time() - compile_started
    result["helion_max_rel_error"] = rel_error(base_y, helion_y)

    helion_ms, helion_y = cuda_time_ms(lambda: helion_silu_and_mul(x), warmup, iters)
    result["helion_ms"] = helion_ms
    result["helion_speedup_vs_torch"] = base_ms / helion_ms if helion_ms else None
    result["helion_max_rel_error_after_timing"] = rel_error(base_y, helion_y)

    if hf_activation is not None:
        def hf_silu_and_mul() -> torch.Tensor:
            hf_activation.silu_and_mul(out, x)
            return out

        hf_ms, hf_y = cuda_time_ms(hf_silu_and_mul, warmup, iters)
        result["hf_activation_ms"] = hf_ms
        result["hf_activation_speedup_vs_torch"] = base_ms / hf_ms if hf_ms else None
        result["hf_activation_max_rel_error"] = rel_error(base_y, hf_y)

    return result


def write_markdown(results: list[dict], path: Path) -> None:
    lines = ["# Helion Kernel Lab", ""]
    lines.append("| op | shape | torch ms | helion ms | helion speedup | helion first call s | hf ms | hf speedup |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        lines.append(
            "| silu_and_mul | ({shape0}, {shape1}) | {torch_ms:.4f} | {helion_ms:.4f} | {helion_speedup_vs_torch:.3f} | {helion_first_call_s:.2f} | {hf_ms} | {hf_speedup} |".format(
                shape0=r["shape"][0],
                shape1=r["shape"][1],
                torch_ms=r["torch_ms"],
                helion_ms=r["helion_ms"],
                helion_speedup_vs_torch=r["helion_speedup_vs_torch"],
                helion_first_call_s=r["helion_first_call_s"],
                hf_ms=f"{r['hf_activation_ms']:.4f}" if "hf_activation_ms" in r else "",
                hf_speedup=f"{r['hf_activation_speedup_vs_torch']:.3f}" if "hf_activation_speedup_vs_torch" in r else "",
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Helion first-call time includes compilation and quick autotuning.")
    lines.append("- Compare Helion against both Torch and HF activation because HF is already a strong fused kernel candidate.")
    lines.append("- Autotune logs are written under `/scratch/deepseek-prof/profiles/helion/`.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--shapes", nargs="*", default=["1024x3072", "8192x3072"])
    args = parser.parse_args()

    os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / "cache" / "triton"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / "cache" / "torchinductor"))

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    torch.manual_seed(0)
    results = []
    for spec in args.shapes:
        rows_s, hidden_s = spec.lower().split("x", 1)
        results.append(bench_shape(int(rows_s), int(hidden_s), args.warmup, args.iters))

    json_path = OUT / "helion_kernel_lab.json"
    md_path = OUT / "helion_kernel_lab.md"
    json_path.write_text(json.dumps(results, indent=2))
    write_markdown(results, md_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
