import os
from pathlib import Path

import torch
import torch._dynamo as dynamo
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile


ROOT = Path("/scratch/deepseek-prof")
OUT = ROOT / "profiles" / "torch_dynamo_inductor"
OUT.mkdir(parents=True, exist_ok=True)


class TinyBlock(nn.Module):
    def __init__(self, width: int = 1024) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.w1 = nn.Linear(width, width * 4, bias=False)
        self.w2 = nn.Linear(width * 4, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)
        x = torch.nn.functional.gelu(self.w1(x), approximate="tanh")
        return residual + self.w2(x)


def main() -> None:
    torch.manual_seed(0)
    device = "cuda"
    module = TinyBlock().to(device=device, dtype=torch.float16).eval()
    x = torch.randn((8, 128, 1024), device=device, dtype=torch.float16)

    exported = torch.export.export(module, (x,))
    (OUT / "exported_aten_graph.py").write_text(exported.graph_module.code)

    explain = dynamo.explain(module)(x)
    (OUT / "dynamo_explain.txt").write_text(str(explain))

    compiled = torch.compile(module, backend="inductor", fullgraph=True)

    # First call compiles; following calls exercise cache/steady state.
    with torch.inference_mode():
        for _ in range(3):
            y = compiled(x)
        torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.inference_mode():
            for _ in range(5):
                y = compiled(x)
            torch.cuda.synchronize()

    prof.export_chrome_trace(str(OUT / "torch_profiler_compile_lab.json"))
    (OUT / "torch_profiler_table.txt").write_text(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=50
        )
    )
    print("output_dir", OUT)
    print("result_sum", float(y.float().sum().cpu()))
    print("TORCH_LOGS_OUT", os.environ.get("TORCH_LOGS_OUT"))
    print("TORCHINDUCTOR_CACHE_DIR", os.environ.get("TORCHINDUCTOR_CACHE_DIR"))


if __name__ == "__main__":
    main()
