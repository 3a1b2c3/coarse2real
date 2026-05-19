import inspect
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    timeout_sec = int(os.environ.get("C2R_NCCL_SMOKE_TIMEOUT_SEC", "30"))

    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

    if not torch.cuda.is_available():
        raise RuntimeError("NCCL smoke test requires CUDA.")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    init_kwargs = {
        "backend": "nccl",
        "init_method": "env://",
        "timeout": timedelta(seconds=timeout_sec),
    }
    try:
        if "device_id" in inspect.signature(dist.init_process_group).parameters:
            init_kwargs["device_id"] = device
    except (TypeError, ValueError):
        pass

    dist.init_process_group(**init_kwargs)

    if rank == 0:
        print(
            f"Running NCCL preflight on {world_size} rank(s) with timeout={timeout_sec}s "
            f"(device={device})."
        )

    reduce_tensor = torch.zeros(1, device=device)
    dist.all_reduce(reduce_tensor)

    gather_tensor = torch.full((1, 4, 8), float(rank), device=device, dtype=torch.bfloat16)
    gathered = [torch.empty_like(gather_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, gather_tensor.contiguous())
    torch.cuda.synchronize(device)

    if rank == 0:
        print("NCCL preflight passed.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
