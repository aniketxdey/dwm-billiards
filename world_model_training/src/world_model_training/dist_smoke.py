from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available()

    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    x = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    if rank == 0:
        expected = world_size * (world_size + 1) / 2.0
        print(
            f"dist_smoke_ok=True world_size={world_size} backend={backend} "
            f"sum={x.item():.1f} expected={expected:.1f}"
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
