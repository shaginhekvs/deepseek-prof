import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    x = torch.ones((256 * 1024 * 1024,), device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    print(f"rank={rank}/{world} local_rank={local_rank} sum0={float(x[0].cpu())}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
