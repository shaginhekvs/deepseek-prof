import torch


def main() -> None:
    print("torch", torch.__version__)
    print("cuda available", torch.cuda.is_available())
    print("cuda runtime", torch.version.cuda)
    device = "cuda:0"
    torch.manual_seed(0)
    x = torch.randn((4096, 4096), device=device, dtype=torch.float16) * 0.01
    w = torch.randn((4096, 4096), device=device, dtype=torch.float16) * 0.01
    y = x
    for _ in range(5):
        y = torch.relu(y @ w)
    torch.cuda.synchronize()
    print("sum", float(y.float().sum().cpu()))


if __name__ == "__main__":
    main()
