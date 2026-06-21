import json
import sys
from pathlib import Path


def main() -> None:
    for name in sys.argv[1:]:
        path = Path(name)
        data = json.loads(path.read_text())
        print(f"\n{path}")
        for key in (
            "elapsed_time",
            "num_requests",
            "total_num_tokens",
            "requests_per_second",
            "tokens_per_second",
            "total_input_tokens",
            "total_output_tokens",
            "request_throughput",
            "total_token_throughput",
            "output_token_throughput",
        ):
            if key in data:
                print(f"{key}: {data[key]}")


if __name__ == "__main__":
    main()
