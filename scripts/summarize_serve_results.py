import json
import sys
from pathlib import Path


FIELDS = [
    "completed",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p90_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p90_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p90_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p90_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
]


def load_result(path: Path) -> dict:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        if not data:
            raise ValueError(f"{path} is an empty list")
        data = data[-1]
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object/list")
    return data


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    rows = []
    for path in sorted(paths):
        if path.name == "summary.md":
            continue
        result = load_result(path)
        row = {"file": path.name}
        for field in FIELDS:
            row[field] = result.get(field)
        rows.append(row)

    if not rows:
        print("No benchmark JSON files found.")
        return

    columns = ["file"] + FIELDS
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")

    print("\n## Bottleneck Reading")
    print("- High `mean_ttft_ms` with low `mean_tpot_ms`: prefill/model-load/scheduler admission pressure.")
    print("- Low TTFT but high `mean_tpot_ms`/`mean_itl_ms`: decode, sampling, KV-cache, or GPU launch overhead.")
    print("- Latency grows sharply from concurrency 1 to 8/32: saturation or queueing.")
    print("- Output throughput plateaus while latency rises: GPU is saturated; inspect Nsight kernels next.")
    print("- Request throughput low but GPU kernels sparse: frontend/tokenization/HTTP/scheduler CPU bottleneck.")


if __name__ == "__main__":
    main()
