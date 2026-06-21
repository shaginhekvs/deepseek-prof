import torch
from torch.profiler import ProfilerActivity, profile


def main() -> None:
    print("torch", torch.__version__)
    print("cuda available", torch.cuda.is_available())
    print("cuda runtime", torch.version.cuda)
    print("gpu count", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
        x = torch.randn((2048, 2048), device=f"cuda:{i}", dtype=torch.float16)
        y = x @ x
        torch.cuda.synchronize(i)
        print("  sum", float(y.float().sum().cpu()))

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        x = torch.randn((4096, 4096), device="cuda:0", dtype=torch.float16)
        for _ in range(5):
            x = torch.relu(x @ x)
        torch.cuda.synchronize()

    trace = "/scratch/deepseek-prof/profiles/torch/smoke/trace.json"
    prof.export_chrome_trace(trace)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    print(f"trace {trace}")


if __name__ == "__main__":
    main()
