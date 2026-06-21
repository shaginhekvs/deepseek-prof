#!/usr/bin/env python3
"""Benchmark local GPU pair topology.

Measures:
* CUDA peer copy latency/bandwidth for selected physical GPU pairs.
* NCCL two-rank all_reduce latency/bandwidth for the same pairs.

The goal is to quantify the practical gap between close pairs such as 0-1 and
cross-NUMA pairs such as 0-2 on this A10 server.
"""

import argparse
import json
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


ROOT = Path("/scratch/deepseek-prof")
OUT = ROOT / "profiles" / "gpu_topology"
OUT.mkdir(parents=True, exist_ok=True)


def parse_size(spec: str) -> int:
    spec = spec.strip().lower()
    units = [("gb", 1024**3), ("mb", 1024**2), ("kb", 1024), ("b", 1)]
    for suffix, mul in units:
        if spec.endswith(suffix):
            return int(float(spec[: -len(suffix)]) * mul)
    return int(spec)


def fmt_size(nbytes: int) -> str:
    for suffix, mul in [("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)]:
        if nbytes >= mul:
            val = nbytes / mul
            return f"{val:g}{suffix}"
    return f"{nbytes}B"


def cuda_copy_time_ms(src_dev: int, dst_dev: int, nbytes: int, warmup: int, iters: int):
    src = torch.empty((nbytes,), device=f"cuda:{src_dev}", dtype=torch.uint8)
    dst = torch.empty((nbytes,), device=f"cuda:{dst_dev}", dtype=torch.uint8)
    src.fill_(7)
    torch.cuda.synchronize(src_dev)
    torch.cuda.synchronize(dst_dev)

    with torch.cuda.device(dst_dev):
        stream = torch.cuda.Stream(device=dst_dev)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                dst.copy_(src, non_blocking=True)
            stream.synchronize()
            start.record(stream)
            for _ in range(iters):
                dst.copy_(src, non_blocking=True)
            end.record(stream)
        end.synchronize()

    return start.elapsed_time(end) / iters


def bench_p2p_pair(pair: tuple[int, int], sizes: list[int]) -> list[dict]:
    rows = []
    src, dst = pair
    for nbytes in sizes:
        if nbytes <= 4 * 1024:
            warmup, iters = 200, 2000
        elif nbytes <= 1024 * 1024:
            warmup, iters = 50, 500
        else:
            warmup, iters = 5, 50
        ms = cuda_copy_time_ms(src, dst, nbytes, warmup, iters)
        gbps = (nbytes / 1e9) / (ms / 1e3)
        rows.append(
            {
                "kind": "cuda_peer_copy",
                "src": src,
                "dst": dst,
                "size_bytes": nbytes,
                "size": fmt_size(nbytes),
                "ms": ms,
                "gbps_one_way": gbps,
            }
        )
    return rows


def free_tcp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def nccl_worker(rank: int, pair: tuple[int, int], nbytes: int, warmup: int, iters: int, port: int, queue):
    device = pair[rank]
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
    )
    # float16 keeps tensor sizes realistic for LLM activations.
    nelems = nbytes // 2
    x = torch.ones((nelems,), device=f"cuda:{device}", dtype=torch.float16)
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    start.record()
    for _ in range(iters):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / iters
    if rank == 0:
        queue.put(ms)
    dist.destroy_process_group()


def bench_nccl_pair(pair: tuple[int, int], sizes: list[int]) -> list[dict]:
    rows = []
    ctx = mp.get_context("spawn")
    for nbytes in sizes:
        if nbytes <= 4 * 1024:
            warmup, iters = 50, 500
        elif nbytes <= 1024 * 1024:
            warmup, iters = 20, 200
        else:
            warmup, iters = 5, 50
        queue = ctx.Queue()
        port = free_tcp_port()
        procs = []
        for rank in range(2):
            p = ctx.Process(
                target=nccl_worker,
                args=(rank, pair, nbytes, warmup, iters, port, queue),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"NCCL worker failed for pair={pair} size={nbytes}")
        ms = queue.get()
        # For two-rank ring all_reduce, bus bandwidth is roughly payload/ms.
        # Algorithmic bandwidth reports useful tensor bytes per rank per second.
        alg_gbps = (nbytes / 1e9) / (ms / 1e3)
        rows.append(
            {
                "kind": "nccl_all_reduce_2gpu",
                "gpus": list(pair),
                "size_bytes": nbytes,
                "size": fmt_size(nbytes),
                "ms": ms,
                "alg_gbps_per_rank": alg_gbps,
            }
        )
    return rows


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = ["# GPU Topology Pair Benchmark", ""]
    lines.append("## CUDA Peer Copy")
    lines.append("")
    lines.append("| pair | size | ms | one-way GB/s |")
    lines.append("|---|---:|---:|---:|")
    for row in rows:
        if row["kind"] != "cuda_peer_copy":
            continue
        lines.append(
            f"| {row['src']}->{row['dst']} | {row['size']} | "
            f"{row['ms']:.4f} | {row['gbps_one_way']:.2f} |"
        )
    lines.append("")
    lines.append("## NCCL All-Reduce")
    lines.append("")
    lines.append("| pair | size | ms | alg GB/s per rank |")
    lines.append("|---|---:|---:|---:|")
    for row in rows:
        if row["kind"] != "nccl_all_reduce_2gpu":
            continue
        pair = ",".join(str(x) for x in row["gpus"])
        lines.append(
            f"| {pair} | {row['size']} | {row['ms']:.4f} | "
            f"{row['alg_gbps_per_rank']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=["0,1", "0,2", "2,3", "1,3"])
    parser.add_argument("--sizes", nargs="+", default=["4KB", "1MB", "64MB", "256MB"])
    parser.add_argument("--skip-nccl", action="store_true")
    args = parser.parse_args()

    pairs = [tuple(int(x) for x in item.split(",")) for item in args.pairs]
    sizes = [parse_size(item) for item in args.sizes]
    rows = []

    for pair in pairs:
        print(f"CUDA peer copy {pair}", flush=True)
        rows.extend(bench_p2p_pair(pair, sizes))
    if not args.skip_nccl:
        for pair in pairs:
            print(f"NCCL all_reduce {pair}", flush=True)
            rows.extend(bench_nccl_pair(pair, sizes))

    json_path = OUT / "gpu_topology_pairs.json"
    md_path = OUT / "gpu_topology_pairs.md"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    write_markdown(rows, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    # Keep NCCL logs out of the benchmark by default unless the caller opts in.
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    main()
